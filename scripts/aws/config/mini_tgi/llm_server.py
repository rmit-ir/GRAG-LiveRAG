#!/usr/bin/env python3
"""
Custom HuggingFace handler that supports both token logits and text generation.
Return logits for specific tokens and generate text.

python -m pip install torch transformers fastapi uvicorn accelerate
python llm_server.py --batch-size 64

curl --request POST \
  --url http://0.0.0.0:8000/logits \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: insomnia/11.0.2' \
  --data '{
	"prompt": "User: Hello, are you there?\nAssistant: ",
	"tokens": ["Yes", "No", "What", "Hello"]
}'
"""

import torch
import logging
import asyncio
import multiprocessing
import threading
import uuid
from typing import Dict, List, TypedDict, Optional, Union, Tuple, Set, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Define request type constants
REQUEST_TYPE_LOGITS = "logits"
REQUEST_TYPE_GENERATION = "generation"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm-server")

# Define typed dictionaries for better type checking
class LogitsResultDict(TypedDict):
    """Result dictionary for token logits."""
    logits: Dict[str, float]
    probabilities: Dict[str, float]
    raw_probabilities: Dict[str, float]
    next_token: str


class GenerationResultDict(TypedDict):
    """Result dictionary for text generation."""
    text: str
    logits: Optional[List[List[float]]]


class ErrorResultDict(TypedDict):
    """Error result dictionary."""
    error: str


class ModelStatusDict(TypedDict):
    """Status dictionary for model status updates."""
    status: str  # "loading", "ready", "error", "shutting_down", "shutdown"
    message: str
    device: Optional[str] = None


# Define request typed dictionaries
class LogitsRequestDict(TypedDict):
    """Request dictionary for logits."""
    type: str  # REQUEST_TYPE_LOGITS
    id: str
    prompt: str
    tokens: List[str]


class GenerationRequestDict(TypedDict):
    """Request dictionary for generation."""
    type: str  # REQUEST_TYPE_GENERATION
    id: str
    prompt: str
    messages: Optional[List[Dict[str, str]]]
    max_tokens: int
    temperature: float
    output_scores: bool


class BatchProcessor:
    """
    Processes batches of requests in a separate process to bypass the GIL.
    """

    def __init__(self, model_id: str, max_batch_size: int = 8):
        """
        Initialize the batch processor.

        Args:
            model_id: ID of the model to load
            max_batch_size: Maximum number of requests in a batch
        """
        self.model_id = model_id
        self.max_batch_size = max_batch_size

        # Queues for communication between processes
        # These queues are created before the worker process and shared through inheritance
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.status_queue = multiprocessing.Queue()  # Queue for model status updates
        self.shutdown_event = multiprocessing.Event()
        
        # Model status tracking
        self.model_ready = False

        # Start the worker process
        self.worker_process = multiprocessing.Process(
            target=self._worker_process_function,
            daemon=True
        )
        self.worker_process.start()

        # Dictionary to store pending requests
        self.pending_requests = {}

        # Start the response handler
        self.response_handler_running = True
        self.loop = asyncio.new_event_loop()
        self.response_handler_thread = threading.Thread(
            target=self._run_response_handler_in_thread,
            daemon=True
        )
        self.response_handler_thread.start()

        logger.info(
            f"Started batch processor with max batch size {max_batch_size}")

    def _run_response_handler_in_thread(self):
        """
        Run the response handler in a separate thread with its own event loop.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._handle_responses())

    async def _handle_responses(self):
        """
        Handle responses from the worker process.
        """
        while self.response_handler_running:
            try:
                # Check if we have any responses
                if not self.response_queue.empty():
                    request_id, result = self.response_queue.get_nowait()

                    # Check if we have a pending request with this ID
                    if request_id in self.pending_requests:
                        # Set the result and remove from pending requests
                        self.pending_requests[request_id].set_result(result)
                        del self.pending_requests[request_id]

                # Sleep a bit to avoid busy waiting
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error handling response: {str(e)}")
                await asyncio.sleep(0.1)  # Sleep longer on error

    def _worker_process_function(self):
        """
        Worker process function that processes batches of requests.
        """
        try:
            # Update status to loading
            loading_status: ModelStatusDict = {
                "status": "loading", 
                "message": f"Loading model {self.model_id}"
            }
            self.status_queue.put(loading_status)
            
            # Load model in the worker process
            logger.info(f"Loading model {self.model_id} in worker process")
            device = "cuda" if torch.cuda.is_available(
            ) else "mps" if torch.backends.mps.is_available() else "cpu"

            # Load tokenizer
            try:
                tokenizer_status: ModelStatusDict = {
                    "status": "loading", 
                    "message": "Loading tokenizer"
                }
                self.status_queue.put(tokenizer_status)
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                error_msg = f"Failed to load tokenizer: {str(e)}"
                logger.error(error_msg)
                error_status: ModelStatusDict = {
                    "status": "error", 
                    "message": error_msg
                }
                self.status_queue.put(error_status)
                raise

            # Load model
            try:
                model_loading_status: ModelStatusDict = {
                    "status": "loading", 
                    "message": "Loading model weights"
                }
                self.status_queue.put(model_loading_status)
                if device == "mps":
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                    )
                    model = model.to(torch.device("mps"))
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        device_map="auto",
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    )
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                logger.error(error_msg)
                model_error_status: ModelStatusDict = {
                    "status": "error", 
                    "message": error_msg
                }
                self.status_queue.put(model_error_status)
                raise

            logger.info(f"Model loaded successfully on {device}")
            
            # Send a message to the parent process that the model is ready
            ready_status: ModelStatusDict = {
                "status": "ready", 
                "device": device, 
                "message": "Model loaded and ready to serve requests"
            }
            self.status_queue.put(ready_status)
            
            # Process batches continuously
            while not self.shutdown_event.is_set():
                batch = []

                # Get the first request (blocking)
                try:
                    request = self.request_queue.get(timeout=1.0)
                    if request is None:  # Shutdown signal
                        self.shutdown_event.set()
                        break
                    batch.append(request)
                except Exception:
                    # Queue.Empty or other exception, try again
                    continue

                # Try to get more requests up to max_batch_size (non-blocking)
                while len(batch) < self.max_batch_size:
                    try:
                        request = self.request_queue.get_nowait()
                        if request is None:  # Shutdown signal
                            self.shutdown_event.set()
                            break
                        batch.append(request)
                    except Exception:
                        # Queue is empty or other exception, process what we have
                        break

                # Process the batch
                if batch:
                    # Group requests by type (logits or generation)
                    logits_requests = []
                    generation_requests = []

                    for req in batch:
                        # Use the 'type' field from the request dictionary
                        request_type = req["type"]
                        if request_type == REQUEST_TYPE_LOGITS:
                            logits_requests.append(req)
                        elif request_type == REQUEST_TYPE_GENERATION:
                            generation_requests.append(req)
                        else:
                            logger.error(
                                f"Unknown request type: {request_type}")

                    # Process logits requests
                    if logits_requests:
                        logger.info(
                            f"Processing batch of {len(logits_requests)} logits requests")
                        self._process_logits_batch(
                            logits_requests, model, tokenizer, device)

                    # Process generation requests
                    if generation_requests:
                        logger.info(
                            f"Processing batch of {len(generation_requests)} generation requests")
                        self._process_generation_batch(
                            generation_requests, model, tokenizer, device)

                    # Clean up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    else:
                        gc.collect()

        except Exception as e:
            error_msg = f"Worker process error: {str(e)}"
            logger.error(error_msg)
            # Update status queue with error
            worker_error_status: ModelStatusDict = {
                "status": "error", 
                "message": error_msg
            }
            self.status_queue.put(worker_error_status)

    def _process_logits_batch(
        self,
        batch: List[LogitsRequestDict],
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        device: Union[str, torch.device]
    ) -> None:
        """
        Process a batch of logits requests with true batching.
        This implementation batches all prompts together regardless of uniqueness,
        allowing for a single forward pass through the model.
        
        Args:
            batch: List of logits request dictionaries
            model: The loaded language model
            tokenizer: The tokenizer for the model
            device: The device to run inference on (cuda, mps, or cpu)
        """
        try:
            # Prepare data structures to track requests and their tokens
            all_prompts: List[str] = []
            request_info: List[Tuple[str, List[str], int]] = []  # Store (request_id, tokens, batch_idx) for each request
            
            # Collect all prompts and tokens
            for idx, req in enumerate(batch):
                prompt: str = req["prompt"]
                request_id: str = req["id"]
                tokens: List[str] = req["tokens"]
                
                all_prompts.append(prompt)
                request_info.append((request_id, tokens, idx))
            
            # Tokenize all prompts in a single batch with padding
            # This ensures all sequences have the same length
            inputs: Dict[str, torch.Tensor] = tokenizer(
                all_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            ).to(device)
            
            # Collect all unique tokens across all requests
            all_tokens: Set[str] = set()
            for _, tokens, _ in request_info:
                all_tokens.update(tokens)
            
            all_tokens_list: List[str] = list(all_tokens)
            token_ids: List[int] = []
            
            # Get token IDs for all tokens
            for token in all_tokens_list:
                token_id: int = tokenizer.encode(token, add_special_tokens=False)[0]
                token_ids.append(token_id)
            
            # Create a mapping from token to token_id for faster lookup
            token_to_id: Dict[str, int] = {token: token_id for token, token_id in zip(all_tokens_list, token_ids)}
            
            # Generate logits for all prompts in a single forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                # Get logits for the last token of each sequence
                # Shape: [batch_size, vocab_size]
                batch_logits: torch.Tensor = outputs.logits[:, -1, :]
            
            # Process each request
            for request_id, tokens, batch_idx in request_info:
                # Get the logits for this specific request
                logits: torch.Tensor = batch_logits[batch_idx]
                
                # Extract logits for the requested tokens
                token_logits: Dict[str, float] = {}
                for token in tokens:
                    token_id = token_to_id[token]
                    token_logits[token] = logits[token_id].item()
                
                # Calculate probabilities for the requested tokens
                item_logits: List[float] = [logits[token_to_id[token]].item() for token in tokens]
                item_logits_tensor: torch.Tensor = torch.tensor(item_logits)
                item_probs: List[float] = torch.nn.functional.softmax(item_logits_tensor, dim=0).tolist()
                token_probs: Dict[str, float] = {token: prob for token, prob in zip(tokens, item_probs)}
                
                # Apply softmax to the entire logits tensor to get raw probabilities
                full_probs: torch.Tensor = torch.nn.functional.softmax(logits, dim=0)
                
                # Find the token with the highest probability
                max_prob_idx: int = torch.argmax(full_probs).item()
                max_prob_token: str = tokenizer.decode([max_prob_idx])
                
                # Extract raw probabilities for the requested tokens
                raw_token_probs: Dict[str, float] = {
                    token: full_probs[token_to_id[token]].item()
                    for token in tokens
                }
                
                # Create result
                result: LogitsResultDict = {
                    "logits": token_logits,
                    "probabilities": token_probs,
                    "raw_probabilities": raw_token_probs,
                    "next_token": max_prob_token
                }
                
                # Send result back through the response queue
                self.response_queue.put((request_id, result))
                
            # Log the performance gain from batching
            if len(all_prompts) > 1:
                logger.info(f"Batch processed logits for {len(all_prompts)} prompts in a single forward pass")
                
        except Exception as e:
            logger.error(f"Error processing logits batch: {str(e)}")
            # Send error to all items in the batch
            error_result: ErrorResultDict = {"error": str(e)}
            for req in batch:
                request_id: str = req["id"]
                self.response_queue.put((request_id, error_result))

    def _process_generation_batch(
        self,
        batch: List[GenerationRequestDict],
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        device: Union[str, torch.device]
    ) -> None:
        """
        Process a batch of generation requests with true batching.
        This implementation groups requests with similar parameters to enable
        batched generation where possible.
        
        Args:
            batch: List of generation request dictionaries
            model: The loaded language model
            tokenizer: The tokenizer for the model
            device: The device to run inference on (cuda, mps, or cpu)
        """
        try:
            # Group requests by similar generation parameters
            # This allows us to batch requests with the same max_tokens, temperature, etc.
            parameter_groups: Dict[Tuple[int, float, bool], List[Tuple[str, str, Optional[List[Dict[str, str]]]]]] = {}
            
            for req in batch:
                request_id: str = req["id"]
                prompt: str = req["prompt"]
                messages: Optional[List[Dict[str, str]]] = req.get("messages")
                max_tokens: int = req["max_tokens"]
                temperature: float = req.get("temperature", 0.7)
                output_scores: bool = req.get("output_scores", False)
                
                # Create a key for grouping similar requests
                param_key = (max_tokens, temperature, output_scores)
                
                if param_key not in parameter_groups:
                    parameter_groups[param_key] = []
                
                parameter_groups[param_key].append((request_id, prompt, messages))
            
            # Process each parameter group
            for param_key, requests in parameter_groups.items():
                max_tokens, temperature, output_scores = param_key
                
                # Process requests with the same parameters in batches
                # For simplicity, we'll process requests with messages separately
                prompts_only: List[Tuple[str, str, None]] = []
                messages_only: List[Tuple[str, str, List[Dict[str, str]]]] = []
                
                for request_id, prompt, messages in requests:
                    if messages:
                        messages_only.append((request_id, prompt, messages))
                    else:
                        prompts_only.append((request_id, prompt, None))
                
                # Process prompts-only batch
                if prompts_only:
                    try:
                        # Extract prompts and request IDs
                        prompt_request_ids: List[str] = [req_id for req_id, _, _ in prompts_only]
                        prompts: List[str] = [prompt for _, prompt, _ in prompts_only]
                        
                        # Tokenize all prompts in a single batch
                        inputs: Dict[str, torch.Tensor] = tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(device)
                        
                        # Generate text for all prompts in a single batch
                        generation_kwargs: Dict[str, Any] = {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            "output_scores": output_scores,
                            "return_dict_in_generate": True
                        }
                        
                        outputs = model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                        
                        # Process each result
                        for i, request_id in enumerate(prompt_request_ids):
                            # Extract response for this request
                            response_ids: torch.Tensor = outputs.sequences[i]
                            response: str = tokenizer.decode(
                                response_ids, skip_special_tokens=True
                            )
                            
                            # Extract only the assistant's response (remove the prompt)
                            original_prompt = prompts[i]
                            content: str = response[len(original_prompt):].strip()
                            
                            # Create result
                            result: GenerationResultDict = {"text": content}
                            
                            # Add logits if requested
                            if output_scores and hasattr(outputs, "scores"):
                                # Convert logits to Python lists (only for the generated tokens)
                                logits_list: List[List[float]] = []
                                for score_tensor in outputs.scores:
                                    # Get the logits for this specific request
                                    token_logits: List[float] = score_tensor[i].detach().cpu().tolist()
                                    logits_list.append(token_logits)
                                result["logits"] = logits_list
                            
                            # Send result back through the response queue
                            self.response_queue.put((request_id, result))
                        
                        logger.info(f"Batch processed {len(prompts)} generation requests")
                        
                    except Exception as e:
                        logger.error(f"Error processing generation batch: {str(e)}")
                        # Send error to all items in the batch
                        error_result: ErrorResultDict = {"error": str(e)}
                        for request_id, _, _ in prompts_only:
                            self.response_queue.put((request_id, error_result))
                
                # Process messages-only requests individually for now
                # Chat templates make batching more complex
                for request_id, prompt, messages in messages_only:
                    try:
                        # Format input using chat template
                        formatted_prompt: str = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Tokenize input
                        inputs: Dict[str, torch.Tensor] = tokenizer(
                            formatted_prompt, return_tensors="pt"
                        ).to(device)
                        
                        # Generate text
                        generation_kwargs: Dict[str, Any] = {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            "output_scores": output_scores,
                            "return_dict_in_generate": True
                        }
                        
                        outputs = model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                        
                        # Extract response
                        response_ids: torch.Tensor = outputs.sequences[0]
                        response: str = tokenizer.decode(
                            response_ids, skip_special_tokens=True
                        )
                        
                        # Extract only the assistant's response (remove the prompt)
                        content: str = response[len(formatted_prompt):].strip()
                        
                        # Create result
                        result: GenerationResultDict = {"text": content}
                        
                        # Add logits if requested
                        if output_scores and hasattr(outputs, "scores"):
                            # Convert logits to Python lists (only for the generated tokens)
                            logits_list: List[List[float]] = []
                            for score_tensor in outputs.scores:
                                # Get the logits for the token with highest probability
                                token_logits: List[float] = score_tensor[0].detach().cpu().tolist()
                                logits_list.append(token_logits)
                            result["logits"] = logits_list
                        
                        # Send result back through the response queue
                        self.response_queue.put((request_id, result))
                        
                    except Exception as e:
                        logger.error(f"Error processing generation request: {str(e)}")
                        error_result: ErrorResultDict = {"error": str(e)}
                        self.response_queue.put((request_id, error_result))
                        
        except Exception as e:
            logger.error(f"Error processing generation batch: {str(e)}")
            # Send error to all items in the batch
            error_result: ErrorResultDict = {"error": str(e)}
            for req in batch:
                request_id: str = req["id"]
                self.response_queue.put((request_id, error_result))

    async def process_logits_request(self, prompt: str, tokens: List[str]) -> Dict:
        """
        Process a logits request and return the result.

        Args:
            prompt: The input prompt
            tokens: List of tokens to get logits for

        Returns:
            Dictionary with logits and probabilities
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())

        # Create a future to wait for the result
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        # Create a structured request dictionary
        request: LogitsRequestDict = {
            "type": REQUEST_TYPE_LOGITS,
            "id": request_id,
            "prompt": prompt,
            "tokens": tokens
        }

        # Add request to the queue
        self.request_queue.put(request)

        # Wait for the result
        try:
            result = await asyncio.wait_for(future, timeout=60.0)
            return result
        except asyncio.TimeoutError:
            # Remove from pending requests
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            return {"error": "Request timed out"}
        except Exception as e:
            # Remove from pending requests
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            return {"error": f"Request failed: {str(e)}"}

    async def process_generation_request(
        self,
        prompt: str,
        messages: List[Dict[str, str]] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        output_scores: bool = False
    ) -> Dict:
        """
        Process a generation request and return the result.

        Args:
            prompt: The input prompt
            messages: List of message dictionaries with role and content
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            output_scores: Whether to output token scores

        Returns:
            Dictionary with generated text and optionally logits
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())

        # Create a future to wait for the result
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        # Create a structured request dictionary
        request: GenerationRequestDict = {
            "type": REQUEST_TYPE_GENERATION,
            "id": request_id,
            "prompt": prompt,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "output_scores": output_scores
        }

        # Add request to the queue
        self.request_queue.put(request)

        # Wait for the result
        try:
            # Longer timeout for generation
            result = await asyncio.wait_for(future, timeout=120.0)
            return result
        except asyncio.TimeoutError:
            # Remove from pending requests
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            return {"error": "Request timed out"}
        except Exception as e:
            # Remove from pending requests
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            return {"error": f"Request failed: {str(e)}"}

    def shutdown(self):
        """
        Shutdown the batch processor.
        """
        logger.info("Shutting down batch processor")
        
        # Update status queue
        shutdown_init_status: ModelStatusDict = {
            "status": "shutting_down", 
            "message": "Shutting down model server"
        }
        self.status_queue.put(shutdown_init_status)

        # Stop the response handler
        self.response_handler_running = False

        # Signal worker to stop
        self.request_queue.put(None)

        # Wait for worker to terminate
        self.worker_process.join(timeout=5.0)
        if self.worker_process.is_alive():
            logger.warning(
                "Worker process did not terminate gracefully, terminating")
            self.worker_process.terminate()
            # Update status queue
            terminate_error_status: ModelStatusDict = {
                "status": "error", 
                "message": "Worker process did not terminate gracefully"
            }
            self.status_queue.put(terminate_error_status)
        else:
            # Update status queue
            shutdown_complete_status: ModelStatusDict = {
                "status": "shutdown", 
                "message": "Model server shutdown complete"
            }
            self.status_queue.put(shutdown_complete_status)

        # Wait for response handler thread to terminate
        self.response_handler_thread.join(timeout=5.0)

        # Close the event loop
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


# For local testing
if __name__ == "__main__":
    import argparse
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from typing import List as PydanticList, Optional as PydanticOptional

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the LLM server with batch processing")
    parser.add_argument(
        "--model", type=str, default='tiiuae/falcon3-10b-instruct', help="Model ID to use")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Maximum batch size")
    args = parser.parse_args()

    # Create FastAPI app
    app = FastAPI(title="LLM API Server with Batch Processing")

    # Initialize batch processor
    batch_processor = BatchProcessor(
        model_id=args.model,
        max_batch_size=args.batch_size
    )

    # Define request models
    class LogitsRequest(BaseModel):
        prompt: str
        tokens: PydanticList[str]

    class Message(BaseModel):
        role: str
        content: str

    class GenerationRequest(BaseModel):
        prompt: PydanticOptional[str] = None
        messages: PydanticOptional[PydanticList[Message]] = None
        max_tokens: int = Field(default=100, ge=1, le=32_000)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        output_scores: bool = False

    @app.post("/logits")
    async def get_logits(request: LogitsRequest):
        """
        Get logits for specific tokens given a prompt.
        """
        result = await batch_processor.process_logits_request(
            prompt=request.prompt,
            tokens=request.tokens
        )
        return JSONResponse(content=result)

    @app.post("/generate")
    async def generate_text(request: GenerationRequest):
        """
        Generate text given a prompt or messages.
        """
        if not request.prompt and not request.messages:
            return JSONResponse(
                status_code=400,
                content={"error": "Either prompt or messages must be provided"}
            )

        result = await batch_processor.process_generation_request(
            prompt=request.prompt or "",
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            output_scores=request.output_scores
        )
        return JSONResponse(content=result)

    # Start a background task to monitor the model status
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(monitor_model_status())
    
    async def monitor_model_status():
        """
        Monitor the model status queue for updates from the worker process.
        """
        while True:
            try:
                # Check if we have any status updates (non-blocking)
                if not batch_processor.status_queue.empty():
                    status: ModelStatusDict = batch_processor.status_queue.get_nowait()
                    status_type = status.get("status")
                    
                    # Log the status update
                    message = status.get("message", "")
                    logger.info(f"Model status update: {status_type} - {message}")
                    
                    # Update model_ready flag based on status
                    if status_type == "ready":
                        batch_processor.model_ready = True
                        logger.info("Model is ready to serve requests")
                    elif status_type in ["error", "shutdown"]:
                        batch_processor.model_ready = False
                    
                # Sleep a bit to avoid busy waiting
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error monitoring model status: {str(e)}")
                await asyncio.sleep(1.0)  # Sleep longer on error
    
    @app.get("/health")
    async def health_check():
        """
        Health check endpoint that verifies if the model is loaded and ready.
        """
        device = "cuda" if torch.cuda.is_available(
        ) else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Check if the worker process is alive
        worker_alive = batch_processor.worker_process.is_alive()
        
        if not worker_alive:
            error_response: ModelStatusDict = {
                "status": "error",
                "message": "Model worker process is not running",
                "device": device
            }
            return JSONResponse(
                status_code=503,
                content={**error_response, "model": args.model}
            )
        
        # Check if the model is ready
        if not batch_processor.model_ready:
            initializing_response: ModelStatusDict = {
                "status": "initializing",
                "message": "Model is still loading",
                "device": device
            }
            return JSONResponse(
                status_code=503,
                content={**initializing_response, "model": args.model}
            )
        
        # If we get here, the model is ready
        ready_response: ModelStatusDict = {
            "status": "ok",
            "message": "Model is loaded and ready to serve requests",
            "device": device
        }
        return {**ready_response, "model": args.model}

    # Shutdown handler
    @app.on_event("shutdown")
    def shutdown_event():
        batch_processor.shutdown()

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
