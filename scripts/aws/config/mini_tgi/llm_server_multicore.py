#!/usr/bin/env python3
"""
Multi-core version of the HuggingFace handler that supports both token logits and text generation.
This version utilizes multiple CPU cores to improve throughput when GPU utilization is high.

python -m pip install torch transformers fastapi uvicorn accelerate
python llm_server_multicore.py --batch-size 64 --num-workers 8

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
import queue
import uuid
import time
import os
from typing import Dict, List, TypedDict, Optional, Union, Tuple, Set, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
# This must be done at the module level before any multiprocessing operations
multiprocessing.set_start_method('spawn', force=True)

# Define request type constants
REQUEST_TYPE_LOGITS = "logits"
REQUEST_TYPE_GENERATION = "generation"
REQUEST_TYPE_TOKENIZE = "tokenize"
REQUEST_TYPE_INFERENCE = "inference"

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


def _tokenizer_worker_process(model_id, worker_id, request_queue, result_queue, shutdown_event):
    """
    Standalone worker function for tokenization to avoid pickling issues.
    
    Args:
        model_id: ID of the model to load tokenizer for
        worker_id: ID of this tokenizer worker
        request_queue: Queue for receiving requests
        result_queue: Queue for sending results
        shutdown_event: Event for signaling shutdown
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Tokenizer worker {worker_id} initialized")
        
        # Process tokenization requests
        while not shutdown_event.is_set():
            try:
                # Get a request from the queue
                request = request_queue.get(timeout=1.0)
                
                if request is None:  # Shutdown signal
                    break
                    
                request_id = request["id"]
                request_type = request["type"]
                
                if request_type == REQUEST_TYPE_LOGITS:
                    # Tokenize for logits request
                    prompt = request["prompt"]
                    tokens = request["tokens"]
                    
                    # Tokenize the prompt
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        return_attention_mask=True
                    )
                    
                    # Get token IDs for requested tokens
                    token_ids = []
                    for token in tokens:
                        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
                        token_ids.append(token_id)
                    
                    # Create result
                    result = {
                        "id": request_id,
                        "type": request_type,
                        "inputs": {k: v.tolist() for k, v in inputs.items()},
                        "tokens": tokens,
                        "token_ids": token_ids
                    }
                    
                    # Put result in the result queue
                    result_queue.put(result)
                    
                elif request_type == REQUEST_TYPE_GENERATION:
                    # Tokenize for generation request
                    prompt = request.get("prompt", "")
                    messages = request.get("messages")
                    max_tokens = request.get("max_tokens", 100)
                    temperature = request.get("temperature", 0.7)
                    output_scores = request.get("output_scores", False)
                    
                    if messages:
                        # Format input using chat template
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Tokenize input
                        inputs = tokenizer(
                            formatted_prompt, 
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        )
                    else:
                        # Tokenize the prompt directly
                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        )
                    
                    # Create result
                    result = {
                        "id": request_id,
                        "type": request_type,
                        "inputs": {k: v.tolist() for k, v in inputs.items()},
                        "original_prompt": prompt if not messages else formatted_prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "output_scores": output_scores
                    }
                    
                    # Put result in the result queue
                    result_queue.put(result)
                
                else:
                    logger.error(f"Unknown request type: {request_type}")
                    error_result = {
                        "id": request_id,
                        "error": f"Unknown request type: {request_type}"
                    }
                    result_queue.put(error_result)
                    
            except queue.Empty:
                # Queue is empty, continue
                continue
            except Exception as e:
                logger.error(f"Error in tokenizer worker {worker_id}: {str(e)}")
                if 'request_id' in locals():
                    error_result = {
                        "id": request_id,
                        "error": str(e)
                    }
                    result_queue.put(error_result)
                    
    except Exception as e:
        logger.error(f"Tokenizer worker {worker_id} error: {str(e)}")


class TokenizationManager:
    """
    Manages tokenization operations in a separate process to parallelize with model inference.
    """
    
    def __init__(self, model_id: str):
        """
        Initialize the tokenization manager.
        
        Args:
            model_id: ID of the model to load tokenizer for
        """
        self.model_id = model_id
        self.request_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.shutdown_event = multiprocessing.Event()
        
        # Start tokenizer processes
        self.num_tokenizers = max(1, min(4, multiprocessing.cpu_count() // 2))
        self.tokenizer_processes = []
        
        for i in range(self.num_tokenizers):
            # Use the standalone function to avoid pickling issues
            process = multiprocessing.Process(
                target=_tokenizer_worker_process,
                args=(self.model_id, i, self.request_queue, self.result_queue, self.shutdown_event),
                daemon=True
            )
            process.start()
            self.tokenizer_processes.append(process)
            
        logger.info(f"Started {self.num_tokenizers} tokenizer processes")
    
    # The tokenizer worker function has been moved outside the class to avoid pickling issues
    
    def tokenize(self, request):
        self.request_queue.put(request)
    
    def get_result(self, timeout=None):
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def shutdown(self):
        """
        Shutdown the tokenization manager.
        """
        logger.info("Shutting down tokenization manager")
        self.shutdown_event.set()
        
        # Send shutdown signal to all tokenizer processes
        for _ in range(self.num_tokenizers):
            self.request_queue.put(None)
        
        # Wait for all tokenizer processes to terminate
        for i, process in enumerate(self.tokenizer_processes):
            process.join(timeout=5.0)
            if process.is_alive():
                logger.warning(f"Tokenizer process {i} did not terminate gracefully, terminating")
                process.terminate()


def _inference_worker_process(model_id, worker_id, gpu_id, request_queue, result_queue, status_queue, shutdown_event):
    """
    Standalone worker function for inference to avoid pickling issues.
    
    Args:
        model_id: ID of the model to load
        worker_id: ID of this worker
        gpu_id: ID of the GPU to use (-1 for CPU)
        request_queue: Queue for receiving requests
        result_queue: Queue for sending results
        status_queue: Queue for sending status updates
        shutdown_event: Event for signaling shutdown
    """
    try:
        # Set CUDA_VISIBLE_DEVICES environment variable for this process
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Worker {worker_id} set CUDA_VISIBLE_DEVICES={gpu_id}")
        
        # Set process name for better monitoring
        try:
            import setproctitle
            setproctitle.setproctitle(f"llm_inference_{worker_id}_gpu_{gpu_id}")
        except ImportError:
            pass
            
        # Update status to loading
        loading_status = {
            "status": "loading", 
            "message": f"Loading model {model_id} in worker {worker_id} on GPU {gpu_id}"
        }
        status_queue.put(loading_status)
        
        # Set device based on assigned GPU
        # When using CUDA_VISIBLE_DEVICES, the worker process only sees one GPU (device 0)
        if gpu_id >= 0 and torch.cuda.is_available():
            device = "cuda:0"  # Always use cuda:0 when CUDA_VISIBLE_DEVICES is set
            logger.info(f"Worker {worker_id} using GPU {gpu_id} (mapped to cuda:0): {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info(f"Worker {worker_id} using MPS device")
        else:
            device = "cpu"
            logger.info(f"Worker {worker_id} using CPU")
        
        # Load model
        logger.info(f"Loading model {model_id} in worker {worker_id} on {device}")
        
        try:
            if device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                )
                model = model.to(torch.device("mps"))
            elif device.startswith("cuda"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",  # Will map to the only visible GPU
                    torch_dtype=torch.float16,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                )
                model = model.to(torch.device("cpu"))
        except Exception as e:
            error_msg = f"Failed to load model in worker {worker_id}: {str(e)}"
            logger.error(error_msg)
            error_status = {
                "status": "error", 
                "message": error_msg
            }
            status_queue.put(error_status)
            raise
            
        logger.info(f"Model loaded successfully on {device} in worker {worker_id}")
        
        # Send ready status
        ready_status = {
            "status": "ready", 
            "device": device, 
            "message": f"Model ready in worker {worker_id}"
        }
        status_queue.put(ready_status)
        
        # Process requests
        while not shutdown_event.is_set():
            try:
                # Get a request from the queue
                request = request_queue.get(timeout=1.0)
                
                if request is None:  # Shutdown signal
                    break
                    
                request_id = request["id"]
                request_type = request["type"]
                
                # Convert inputs back to tensors and move to device
                inputs = {k: torch.tensor(v).to(device) for k, v in request["inputs"].items()}
                
                if request_type == REQUEST_TYPE_LOGITS:
                    # Process logits request
                    tokens = request["tokens"]
                    token_ids = request["token_ids"]
                    
                    try:
                        # Generate logits
                        with torch.no_grad():
                            outputs = model(**inputs)
                            # Get logits for the last token
                            logits = outputs.logits[0, -1, :]
                        
                        # Extract logits for the requested tokens
                        token_logits = {}
                        for token, token_id in zip(tokens, token_ids):
                            token_logits[token] = logits[token_id].item()
                        
                        # Calculate probabilities
                        item_logits = [logits[token_id].item() for token_id in token_ids]
                        item_logits_tensor = torch.tensor(item_logits)
                        item_probs = torch.nn.functional.softmax(item_logits_tensor, dim=0).tolist()
                        token_probs = {token: prob for token, prob in zip(tokens, item_probs)}
                        
                        # Apply softmax to the entire logits tensor
                        full_probs = torch.nn.functional.softmax(logits, dim=0)
                        
                        # Find the token with the highest probability
                        max_prob_idx = torch.argmax(full_probs).item()
                        
                        # Create tokenizer just for decoding
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        max_prob_token = tokenizer.decode([max_prob_idx])
                        
                        # Extract raw probabilities
                        raw_token_probs = {
                            token: full_probs[token_id].item()
                            for token, token_id in zip(tokens, token_ids)
                        }
                        
                        # Create result
                        result = {
                            "logits": token_logits,
                            "probabilities": token_probs,
                            "raw_probabilities": raw_token_probs,
                            "next_token": max_prob_token
                        }
                        
                        # Send result
                        result_queue.put((request_id, result))
                        
                    except Exception as e:
                        logger.error(f"Error processing logits request in worker {worker_id}: {str(e)}")
                        error_result = {"error": str(e)}
                        result_queue.put((request_id, error_result))
                
                elif request_type == REQUEST_TYPE_GENERATION:
                    # Process generation request
                    max_tokens = request["max_tokens"]
                    temperature = request["temperature"]
                    output_scores = request["output_scores"]
                    original_prompt = request["original_prompt"]
                    
                    try:
                        # Generate text
                        generation_kwargs = {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            "output_scores": output_scores,
                            "return_dict_in_generate": True
                        }
                        
                        outputs = model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                        
                        # Create tokenizer just for decoding
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        
                        # Extract response
                        response_ids = outputs.sequences[0]
                        response = tokenizer.decode(
                            response_ids, skip_special_tokens=True
                        )
                        
                        # Extract only the assistant's response
                        content = response[len(original_prompt):].strip()
                        
                        # Create result
                        result = {"text": content}
                        
                        # Add logits if requested
                        if output_scores and hasattr(outputs, "scores"):
                            # Convert logits to Python lists
                            logits_list = []
                            for score_tensor in outputs.scores:
                                token_logits = score_tensor[0].detach().cpu().tolist()
                                logits_list.append(token_logits)
                            result["logits"] = logits_list
                        
                        # Send result
                        result_queue.put((request_id, result))
                        
                    except Exception as e:
                        logger.error(f"Error processing generation request in worker {worker_id}: {str(e)}")
                        error_result = {"error": str(e)}
                        result_queue.put((request_id, error_result))
                
                else:
                    logger.error(f"Unknown request type: {request_type}")
                    error_result = {"error": f"Unknown request type: {request_type}"}
                    result_queue.put((request_id, error_result))
                
                # Clean up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                else:
                    gc.collect()
                    
            except queue.Empty:
                # Queue is empty, continue
                continue
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {str(e)}")
                if 'request_id' in locals():
                    error_result = {"error": str(e)}
                    result_queue.put((request_id, error_result))
                    
    except Exception as e:
        error_msg = f"Worker {worker_id} process error: {str(e)}"
        logger.error(error_msg)
        worker_error_status = {
            "status": "error", 
            "message": error_msg
        }
        status_queue.put(worker_error_status)


class InferenceWorker:
    """
    Handles model inference in a separate process, assigned to a specific GPU if available.
    """
    
    def __init__(self, model_id: str, worker_id: int, gpu_id: int, request_queue, result_queue, status_queue, shutdown_event):
        """
        Initialize the inference worker.
        
        Args:
            model_id: ID of the model to load
            worker_id: ID of this worker
            gpu_id: ID of the GPU to use (-1 for CPU)
            request_queue: Queue for receiving requests
            result_queue: Queue for sending results
            status_queue: Queue for sending status updates
            shutdown_event: Event for signaling shutdown
        """
        self.model_id = model_id
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.status_queue = status_queue
        self.shutdown_event = shutdown_event
        
        # Start the worker process with the correct GPU environment
        # We need to set CUDA_VISIBLE_DEVICES before the process starts
        # Use the standalone function to avoid pickling issues
        self.process = multiprocessing.Process(
            target=_inference_worker_process,
            args=(model_id, worker_id, gpu_id, request_queue, result_queue, status_queue, shutdown_event),
            daemon=True
        )
        self.process.start()
    
    def shutdown(self):
        """
        Shutdown the inference worker.
        """
        logger.info(f"Shutting down inference worker {self.worker_id}")
        self.request_queue.put(None)
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            logger.warning(f"Worker {self.worker_id} did not terminate gracefully, terminating")
            self.process.terminate()


class MultiProcessBatchProcessor:
    """
    Processes batches of requests using multiple processes, with inference workers assigned to available GPUs.
    """
    
    def __init__(self, model_id: str, max_batch_size: int = 8, num_workers: int = None, gpu_only: bool = False):
        """
        Initialize the batch processor with multiple workers.
        
        Args:
            model_id: ID of the model to load
            max_batch_size: Maximum number of requests in a batch
            num_workers: Number of worker processes (default: one per GPU, or auto-detect if no GPUs)
            gpu_only: If True, only use GPUs and do not fall back to CPU
        """
        self.model_id = model_id
        self.max_batch_size = max_batch_size
        self.gpu_only = gpu_only
        
        # Detect available GPUs
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"Detected {self.gpu_count} GPUs")
            for i in range(self.gpu_count):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.gpu_count = 0
            logger.info("No GPUs detected, using CPU")
        
        # Determine number of inference workers based on available GPUs
        if num_workers is None:
            if self.gpu_count > 0:
                # Default to one inference worker per GPU
                self.num_workers = self.gpu_count
            else:
                # If no GPUs, use half of CPU cores but at least 1
                self.num_workers = max(1, min(4, multiprocessing.cpu_count() // 2))
        else:
            self.num_workers = max(1, num_workers)
            
        logger.info(f"Initializing batch processor with {self.num_workers} inference workers")
        
        # Queues for communication
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.status_queue = multiprocessing.Queue()
        self.inference_queues = [multiprocessing.Queue() for _ in range(self.num_workers)]
        self.shutdown_event = multiprocessing.Event()
        
        # Initialize tokenization manager
        self.tokenization_manager = TokenizationManager(model_id)
        
        # Start inference workers, assigning each to a GPU if available
        self.inference_workers = []
        
        if self.gpu_count > 0:
            # We have GPUs available
            for i in range(self.num_workers):
                # Assign GPU ID: if we have enough GPUs, each worker gets its own
                # If we have fewer GPUs than workers, distribute workers across GPUs
                gpu_id = i % self.gpu_count
                
                worker = InferenceWorker(
                    model_id=model_id,
                    worker_id=i,
                    gpu_id=gpu_id,
                    request_queue=self.inference_queues[i],
                    result_queue=self.response_queue,
                    status_queue=self.status_queue,
                    shutdown_event=self.shutdown_event
                )
                self.inference_workers.append(worker)
                logger.info(f"Worker {i} assigned to GPU {gpu_id}")
        elif not self.gpu_only:
            # No GPUs, fall back to CPU if allowed
            for i in range(self.num_workers):
                worker = InferenceWorker(
                    model_id=model_id,
                    worker_id=i,
                    gpu_id=-1,  # -1 indicates CPU
                    request_queue=self.inference_queues[i],
                    result_queue=self.response_queue,
                    status_queue=self.status_queue,
                    shutdown_event=self.shutdown_event
                )
                self.inference_workers.append(worker)
                logger.info(f"Worker {i} assigned to CPU")
        else:
            # GPU-only mode but no GPUs available
            raise RuntimeError("GPU-only mode specified but no GPUs are available")
        
        # Model status tracking
        self.model_ready = False
        self.workers_ready = 0
        
        # Dictionary to store pending requests
        self.pending_requests = {}
        
        # Start the request dispatcher
        self.dispatcher_running = True
        self.dispatcher_thread = threading.Thread(
            target=self._run_request_dispatcher,
            daemon=True
        )
        self.dispatcher_thread.start()
        
        # Start the response handler
        self.response_handler_running = True
        self.loop = asyncio.new_event_loop()
        self.response_handler_thread = threading.Thread(
            target=self._run_response_handler_in_thread,
            daemon=True
        )
        self.response_handler_thread.start()
        
        logger.info(f"Started batch processor with max batch size {max_batch_size} and {self.num_workers} inference workers")
    
    def _run_request_dispatcher(self):
        """
        Run the request dispatcher in a separate thread.
        This thread dispatches incoming requests to tokenization and then to inference workers.
        """
        next_worker = 0  # For round-robin worker assignment
        
        while self.dispatcher_running:
            try:
                # Step 1: Get incoming requests and send to tokenization
                while not self.request_queue.empty():
                    request = self.request_queue.get_nowait()
                    if request is None:  # Shutdown signal
                        self.shutdown_event.set()
                        break
                    
                    # Send to tokenization manager
                    self.tokenization_manager.tokenize(request)
                
                # Step 2: Get tokenized requests and send to inference workers
                tokenized_result = self.tokenization_manager.get_result(timeout=0.01)
                if tokenized_result:
                    if "error" in tokenized_result:
                        # Handle tokenization error
                        request_id = tokenized_result["id"]
                        error_result = {"error": tokenized_result["error"]}
                        self.response_queue.put((request_id, error_result))
                    else:
                        # Send to next worker in round-robin fashion
                        self.inference_queues[next_worker].put(tokenized_result)
                        next_worker = (next_worker + 1) % self.num_workers
                
                # Sleep a bit to avoid busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in request dispatcher: {str(e)}")
                time.sleep(0.1)  # Sleep longer on error
    
    def _run_response_handler_in_thread(self):
        """
        Run the response handler in a separate thread with its own event loop.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._handle_responses())
    
    async def _handle_responses(self):
        """
        Handle responses from the worker processes.
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
                
                # Check for status updates
                if not self.status_queue.empty():
                    status = self.status_queue.get_nowait()
                    status_type = status.get("status")
                    
                    # Log the status update
                    message = status.get("message", "")
                    logger.info(f"Model status update: {status_type} - {message}")
                    
                    # Update model_ready flag based on status
                    if status_type == "ready":
                        self.workers_ready += 1
                        if self.workers_ready >= self.num_workers:
                            self.model_ready = True
                            logger.info("All workers are ready to serve requests")
                    elif status_type in ["error", "shutdown"]:
                        # If any worker has an error, mark the model as not ready
                        self.model_ready = False
                
                # Sleep a bit to avoid busy waiting
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error handling response: {str(e)}")
                await asyncio.sleep(0.1)  # Sleep longer on error
    
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
        request = {
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
        request = {
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
        shutdown_init_status = {
            "status": "shutting_down", 
            "message": "Shutting down model server"
        }
        self.status_queue.put(shutdown_init_status)
        
        # Stop the dispatcher and response handler
        self.dispatcher_running = False
        self.response_handler_running = False
        
        # Signal shutdown
        self.shutdown_event.set()
        self.request_queue.put(None)
        
        # Shutdown tokenization manager
        self.tokenization_manager.shutdown()
        
        # Shutdown all inference workers
        for worker in self.inference_workers:
            worker.shutdown()
        
        # Wait for threads to terminate
        self.dispatcher_thread.join(timeout=5.0)
        self.response_handler_thread.join(timeout=5.0)
        
        # Close the event loop
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Update status queue
        shutdown_complete_status = {
            "status": "shutdown", 
            "message": "Model server shutdown complete"
        }
        self.status_queue.put(shutdown_complete_status)


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
        description="Run the multi-core LLM server with parallel processing")
    parser.add_argument(
        "--model", type=str, default='tiiuae/falcon3-10b-instruct', help="Model ID to use")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Maximum batch size")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of inference worker processes (default: one per GPU, or auto-detect if no GPUs)")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Only use GPUs, do not fall back to CPU")
    args = parser.parse_args()

    # Create FastAPI app
    app = FastAPI(title="Multi-Core LLM API Server")

    # Initialize batch processor
    batch_processor = MultiProcessBatchProcessor(
        model_id=args.model,
        max_batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu_only=args.gpu_only
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
        Monitor the model status queue for updates from the worker processes.
        """
        while True:
            try:
                # Check if we have any status updates (non-blocking)
                if not batch_processor.status_queue.empty():
                    status = batch_processor.status_queue.get_nowait()
                    status_type = status.get("status")
                    
                    # Log the status update
                    message = status.get("message", "")
                    logger.info(f"Model status update: {status_type} - {message}")
                    
                    # Update model_ready flag based on status
                    if status_type == "ready":
                        batch_processor.workers_ready += 1
                        if batch_processor.workers_ready >= batch_processor.num_workers:
                            batch_processor.model_ready = True
                            logger.info("All workers are ready to serve requests")
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
        
        # Check if all workers are alive
        workers_alive = all(worker.process.is_alive() for worker in batch_processor.inference_workers)
        
        if not workers_alive:
            error_response = {
                "status": "error",
                "message": "One or more model worker processes are not running",
                "device": device
            }
            return JSONResponse(
                status_code=503,
                content={**error_response, "model": args.model}
            )
        
        # Check if the model is ready
        if not batch_processor.model_ready:
            initializing_response = {
                "status": "initializing",
                "message": f"Model is still loading ({batch_processor.workers_ready}/{batch_processor.num_workers} workers ready)",
                "device": device
            }
            return JSONResponse(
                status_code=503,
                content={**initializing_response, "model": args.model}
            )
        
        # If we get here, the model is ready
        ready_response = {
            "status": "ok",
            "message": f"Model is loaded and ready with {batch_processor.num_workers} workers",
            "device": device,
            "workers": batch_processor.num_workers,
            "tokenizers": batch_processor.tokenization_manager.num_tokenizers
        }
        return {**ready_response, "model": args.model}

    @app.get("/stats")
    async def get_stats():
        """
        Get server statistics.
        """
        import psutil
        
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=0.1, percpu=True),
            "memory_percent": psutil.virtual_memory().percent,
            "workers": batch_processor.num_workers,
            "tokenizers": batch_processor.tokenization_manager.num_tokenizers,
            "model_ready": batch_processor.model_ready,
            "workers_ready": batch_processor.workers_ready
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            try:
                gpu_stats = []
                for i in range(torch.cuda.device_count()):
                    gpu_stats.append({
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": torch.cuda.memory_allocated(i) / (1024 ** 3),  # GB
                        "memory_reserved": torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
                    })
                stats["gpu"] = gpu_stats
            except Exception as e:
                stats["gpu_error"] = str(e)
        
        return stats

    # Shutdown handler
    @app.on_event("shutdown")
    def shutdown_event():
        batch_processor.shutdown()

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
