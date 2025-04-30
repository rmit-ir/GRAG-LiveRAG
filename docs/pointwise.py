from typing import List, Tuple
import transformers
from .rankers import LlmRanker, SearchResult
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from .pairwise import Text2TextGenerationDataset
import torch
from tqdm import tqdm


def prompt_generator_t5(prompt_type, original_prompt_number, instruction, output, tone, order, position, role):
    # gives a list of information [0] is flag, [1] is content, which is prompt_package
    if prompt_type == "original":
        original_prompt_dict = {
            1: ["output_1",
                  "Query: {query}\n Passage: {text}\n Does the passage answer the query?  Answer 'Yes' or 'No'"],

            2: ["output_2", "Passage: {text}\n Query: {query}\n "
                              "Is this passage relevant to the query? Please answer True/False. Answer:"],

            3: ["output_3",
                  "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat "
                  "Relevant', or 'Not Relevant'.\n"
                  "Query: {query} Document:{text} Output:"],

            4: ["output_4", "From a scale of 0 to 4, judge the relevance between the query and the document.'.\n "
                              "Query: {query} Document:{text} Output:"],
        }
        return original_prompt_dict[original_prompt_number]

    elif prompt_type == "adjusted":
        instruction_dict = {
            "instruction_1": "Does the passage answer the query?\n",
            "instruction_2": "Is this passage relevant to the query?\n",
            "instruction_3": "For the following query and document, judge whether they are relevant.\n",
            "instruction_4": "Judge the relevance between the query and the document.\n"
        }

        output_dict = {
            "output_1": "Answer 'Yes' or 'No'. ",
            "output_2": "Answer True/False. ",
            "output_3": "Judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'. ",
            "output_4": "From a scale of 0 to 4, judge the relevance between the query and the document."
        }

        tone_dict = {
            "tone_0": "",
            "tone_1": "Please ",
            "tone_2": "Only ",
            "tone_3": "Must ",
            "tone_4": "You better get this right or you will be punished. ",
            "tone_5": "Only response the ranking results, do not say any word or explain. "
        }

        role_dict = {
            "True": "You are RankGPT, "
                    "an intelligent assistant that can rank passages based on their relevancy to the query. \n",
            "False": ""
        }

        prompt_instruction = instruction_dict[instruction] + "Query: {query}\n"
        passages = "\nPassage: {text}\n"

        if order == "query_first":
            if position == "beginning":
                prompt = role_dict[role] + prompt_instruction + passages + tone_dict[tone] + output_dict[output]
            elif position == "ending":
                prompt = role_dict[role] + tone_dict[tone] + output_dict[output] + prompt_instruction + passages
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        elif order == "passage_first":
            if position == "beginning":
                prompt = role_dict[role] + passages + prompt_instruction + tone_dict[tone] + output_dict[output]
            elif position == "ending":
                prompt = role_dict[role] + tone_dict[tone] + output_dict[output] + passages + prompt_instruction
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        else:
            raise NotImplementedError(f"Order {order} is not implemented, position is {position}.")

        prompt_package = [output, prompt]

        return prompt_package


def prompt_generator_mistral(prompt_type, original_prompt_number, instruction, output, tone, order, position, role):
    # gives a list of information [0] is flag, [1] is content, which is prompt_package
    if prompt_type == "original":
        original_prompt_dict = {
            1: ["output_1",
                  "Query: {query}\n Passage: {text}\n Does the passage answer the query?  Answer 'Yes' or 'No'"],

            2: ["output_2", "Passage: {text}\n Query: {query}\n "
                              "Is this passage relevant to the query? Please answer True/False. Answer:"],

            3: ["output_3",
                  "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat "
                  "Relevant', or 'Not Relevant'.\n"
                  "Query: {query} Document:{text} Output:"],

            4: ["output_4", "From a scale of 0 to 4, judge the relevance between the query and the document.'.\n "
                              "Query: {query} Document:{text} Output:"],
        }

        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"

        role_text = ""
        sys_format = prefix + "{{system}}\n" + role_text + suffix
        prompt = prefix + "{{user}}\n" + original_prompt_dict[original_prompt_number][1] + suffix
        prompt_tem = "<s>[INST] " + sys_format + prompt + "[/INST]</s>"

        prompt_package = [original_prompt_dict[original_prompt_number][0], prompt_tem]

        return prompt_package

    elif prompt_type == "adjusted":
        instruction_dict = {
            "instruction_1": "Does the passage answer the query?\n",
            "instruction_2": "Is this passage relevant to the query?\n",
            "instruction_3": "For the following query and document, judge whether they are relevant.\n",
            "instruction_4": "Judge the relevance between the query and the document.\n"
        }

        output_dict = {
            "output_1": "Answer 'Yes' or 'No'. ",
            "output_2": "Answer True/False. ",
            "output_3": "Judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'. ",
            "output_4": "From a scale of 0 to 4, judge the relevance between the query and the document."
        }

        tone_dict = {
            "tone_0": "",
            "tone_1": "Please ",
            "tone_2": "Only ",
            "tone_3": "Must ",
            "tone_4": "You better get this right or you will be punished. ",
            "tone_5": "Only response the ranking results, do not say any word or explain. "
        }

        role_dict = {
            "True": "You are RankGPT, "
                    "an intelligent assistant that can rank passages based on their relevancy to the query. \n",
            "False": ""
        }

        prompt_instruction = instruction_dict[instruction] + "Query: {query}\n"
        passages = "\nPassage: {text}\n"

        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"

        sys_format = prefix + "{{{system}}\n" + role_dict[role] + suffix

        if order == "query_first":
            if position == "beginning":
                prompt = prefix + "{{{user}}\n" + prompt_instruction + passages + tone_dict[tone] + output_dict[
                    output] + suffix
            elif position == "ending":
                prompt = prefix + "{{{user}}\n" + tone_dict[tone] + output_dict[
                    output] + prompt_instruction + passages + suffix
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        elif order == "passage_first":
            if position == "beginning":
                prompt = prefix + "{{{user}}\n" + passages + prompt_instruction + tone_dict[tone] + output_dict[
                    output] + suffix
            elif position == "ending":
                prompt = prefix + "{{{user}}\n" + tone_dict[tone] + output_dict[
                    output] + passages + prompt_instruction + suffix
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        else:
            raise NotImplementedError(f"Order {order} is not implemented, position is {position}.")

        prompt_tem = "<s>[INST] " + sys_format + prompt + "[/INST]</s>"

        prompt_package = [output, prompt_tem]

        return prompt_package


class PointwiseLlmRanker(LlmRanker):



    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 method="qlm",
                 batch_size=1,
                 cache_dir=None,
                 hf_token=None,
                 prompt_type="adjusted",
                 original_prompt_number=1,
                 instruction="instruction_1",
                 output="output_1",
                 tone="tone_1",
                 order="query_first",
                 position="beginning",
                 role="False"):

        self.HF_TOKEN = hf_token
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, token=self.HF_TOKEN)

        if self.config.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                         if tokenizer_name_or_path is not None else
                                                         model_name_or_path,
                                                         cache_dir=cache_dir)
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
        elif self.config.model_type == 'mistral':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map='auto',
                                                            torch_dtype=torch.float16 if device == 'cuda'
                                                            else torch.float32,
                                                            cache_dir=cache_dir)
            self.tokenizer.use_default_system_prompt = False
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "left"
        elif self.config.model_type == 'llama':
            # self.pipeline = transformers.pipeline(
            #     "text-generation",
            #     model=model_id,
            #     model_kwargs={"torch_dtype": torch.bfloat16},
            #     device="auto",
            # )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                           token=self.HF_TOKEN,
                                                           cache_dir=cache_dir)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            token=self.HF_TOKEN,
                                                            device_map='auto',
                                                            torch_dtype=torch.bfloat16,
                                                            cache_dir=cache_dir)
            self.tokenizer.use_default_system_prompt = False
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.tokenizer.padding_side = "left"

        elif self.config.model_type == 'gemma':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                           token=self.HF_TOKEN,
                                                           cache_dir=cache_dir)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            token=self.HF_TOKEN,
                                                            device_map='auto',
                                                            torch_dtype=torch.float16 if device == 'cuda'
                                                            else torch.float32,
                                                            revision="float16",
                                                            cache_dir=cache_dir)

        self.device = device
        self.method = method
        self.batch_size = batch_size

        # used for prompts
        self.prompt_type = prompt_type
        self.original_prompt_number = original_prompt_number
        self.instruction = instruction
        self.output = output
        self.tone = tone
        self.order = order
        self.position = position
        self.role = role
        self.has_printed_input_text = False

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:

        if self.method == "qlm":
            prompt = "Passage: {text}\nPlease write a question based on this passage."
            print('1')
            data = [prompt.format(text=doc.text) for doc in ranking]
            print('2')
            dataset = Text2TextGenerationDataset(data, self.tokenizer)
            print('3')
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=DataCollatorWithPadding(
                    self.tokenizer,
                    max_length=512,
                    padding='longest',
                ),
                shuffle=False,
                drop_last=False,
                num_workers=4
            )

            labels = self.tokenizer.encode(f"<pad> {query}",
                                           return_tensors="pt",
                                           add_special_tokens=False).to(self.llm.device).repeat(self.batch_size, 1)
            current_id = 0
            with torch.no_grad():
                for batch_inputs in tqdm(loader):

                    batch_labels = labels if labels.shape[0] == len(batch_inputs['input_ids']) \
                        else labels[:len(batch_inputs['input_ids']), :]  # last batch might be smaller

                    batch_inputs = batch_inputs.to(self.llm.device)
                    logits = self.llm(input_ids=batch_inputs['input_ids'],
                                      attention_mask=batch_inputs['attention_mask'],
                                      labels=batch_labels).logits

                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    scores = loss_fct(logits.view(-1, logits.size(-1)), batch_labels.view(-1))
                    scores = -1 * scores.view(-1, batch_labels.size(-1)).sum(dim=1)  # neg log prob
                    for score in scores:
                        ranking[current_id].score = score.item()
                        current_id += 1

        elif self.method == "yes_no":

            if self.config.model_type == 't5':

                # prompt_package[0] is a list that contain 2 elements
                # prompt_package[0] is a flag that shows which type of output is using, and this information is used for counting scores
                # prompt_package[1] is the prompt content
                prompt_package = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                     self.output, self.tone, self.order, self.position, self.role)

                prompt = prompt_package[1]  # pass prompt content

                data = [prompt.format(text=doc.text, query=query) for doc in ranking]

                if not self.has_printed_input_text:  # print a prompt for checking with configuration
                    print("prompt is \n" + data[0] + '\n')
                    # Set the flag to True after printing, to prevent future prints
                    self.has_printed_input_text = True

                dataset = Text2TextGenerationDataset(data, self.tokenizer)

                if prompt_package[0] == "output_1":
                    yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_2":
                    yes_id = self.tokenizer.encode("True", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("False", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_3":
                    yes_id = self.tokenizer.encode("Highly", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("Not", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_4":
                    yes_id = 314  # 4 = 314
                    no_id = 632  # 0 = 632
                else:
                    raise ValueError(f"Invalid type of output {prompt_package[0]}")

                loader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    collate_fn=DataCollatorWithPadding(
                                        self.tokenizer,
                                        # max_length=512,
                                        max_length=2048,
                                        padding='longest'),
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=0
                                    )

                decoder_input_ids = torch.Tensor([self.tokenizer.pad_token_id]).to(self.llm.device,
                                                                                   dtype=torch.long).repeat(
                    self.batch_size,
                    1)

                current_id = 0
                with torch.no_grad():
                    for batch_inputs in tqdm(loader):
                        batch_inputs = batch_inputs.to(self.llm.device)

                        batch_decoder_input_ids = decoder_input_ids if decoder_input_ids.shape[0] == len(
                            batch_inputs['input_ids']) \
                            else decoder_input_ids[:len(batch_inputs['input_ids']),
                                 :]  # last batch might be smaller

                        logits = self.llm(input_ids=batch_inputs['input_ids'],
                                          attention_mask=batch_inputs['attention_mask'],
                                          decoder_input_ids=batch_decoder_input_ids).logits

                        yes_scores = logits[:, :, yes_id]
                        no_scores = logits[:, :, no_id]
                        batch_scores = torch.cat((yes_scores, no_scores), dim=1)
                        batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)

                        scores = batch_scores[:, 0]
                        for score in scores:
                            ranking[current_id].score = score.item()
                            current_id += 1

            elif self.config.model_type == 'mistral':

                prompt_package = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                     self.output, self.tone, self.order, self.position, self.role)

                prompt_tem = prompt_package[1]  # pass prompt content

                conversation = [{"role": "user", "content": prompt_tem}]

                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)[3:]

                data = [prompt.format(text=doc.text, query=query) for doc in ranking]

                if not self.has_printed_input_text:  # print a prompt for checking with configuration
                    print("prompt is \n" + data[0] + '\n')
                    # Set the flag to True after printing, to prevent future prints
                    self.has_printed_input_text = True

                print("prompt is \n" + data[0] + '\n')

                dataset = Text2TextGenerationDataset(data, self.tokenizer)

                if prompt_package[0] == "output_1":
                    yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_2":
                    yes_id = self.tokenizer.encode("True", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("False", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_3":
                    yes_id = self.tokenizer.encode("Highly", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("Not", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_4":
                    yes_id = 28781
                    no_id = 28734
                else:
                    raise ValueError(f"Invalid type of output {prompt_package[0]}")

                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    collate_fn=DataCollatorWithPadding(
                        self.tokenizer,
                        # max_length=512,
                        max_length=2048,
                        padding='longest',
                    ),
                    shuffle=False,
                    drop_last=False,
                    num_workers=0
                )

                current_id = 0
                with torch.no_grad():
                    for batch_inputs in tqdm(loader):
                        # from IPython import embed; embed(); exit()
                        batch_inputs = batch_inputs.to(self.llm.device)
                        logits = self.llm(input_ids=batch_inputs['input_ids'],
                                          attention_mask=batch_inputs['attention_mask'])
                        # print(logits)

                        logits = logits.logits[:, -1, :]
                        # print(logits)

                        yes_scores = logits[:, yes_id]
                        no_scores = logits[:, no_id]

                        batch_scores = torch.stack((yes_scores, no_scores), dim=1)
                        batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                        # probs = torch.nn.functional.softmax(logits, dim=-1)
                        # yes_scores = probs[:, yes_id].tolist()
                        # no_scores = probs[:, no_id].tolist()

                        scores = batch_scores[:, 0]
                        for score in scores:
                            ranking[current_id].score = score.item()
                            current_id += 1

            elif self.config.model_type == 'llama':

                prompt_package = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                     self.output, self.tone, self.order, self.position, self.role)

                prompt_tem = prompt_package[1]  # pass prompt content

                conversation = [{"role": "user", "content": prompt_tem}]

                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False,
                                                            add_generation_prompt=True)[17:]

                data = [prompt.format(text=doc.text, query=query) for doc in ranking]

                if not self.has_printed_input_text:  # print a prompt for checking with configuration
                    print("prompt is \n" + data[0] + '\n')
                    # Set the flag to True after printing, to prevent future prints
                    self.has_printed_input_text = True

                print("prompt is \n" + data[0] + '\n')

                dataset = Text2TextGenerationDataset(data, self.tokenizer)

                if prompt_package[0] == "output_1":
                    yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_2":
                    yes_id = self.tokenizer.encode("True", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("False", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_3":
                    yes_id = self.tokenizer.encode("Highly", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("Not", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_4":
                    yes_id = 19
                    no_id = 15
                else:
                    raise ValueError(f"Invalid type of output {prompt_package[0]}")

                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    collate_fn=DataCollatorWithPadding(
                        self.tokenizer,
                        # max_length=512,
                        max_length=2048,
                        padding='longest',
                    ),
                    shuffle=False,
                    drop_last=False,
                    num_workers=0
                )

                current_id = 0
                with torch.no_grad():
                    for batch_inputs in tqdm(loader):
                        # from IPython import embed; embed(); exit()
                        batch_inputs = batch_inputs.to(self.llm.device)
                        logits = self.llm(input_ids=batch_inputs['input_ids'],
                                          attention_mask=batch_inputs['attention_mask'])
                        # print(logits)

                        logits = logits.logits[:, -1, :]
                        # print(logits)

                        yes_scores = logits[:, yes_id]
                        no_scores = logits[:, no_id]

                        batch_scores = torch.stack((yes_scores, no_scores), dim=1)
                        batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                        # probs = torch.nn.functional.softmax(logits, dim=-1)
                        # yes_scores = probs[:, yes_id].tolist()
                        # no_scores = probs[:, no_id].tolist()

                        scores = batch_scores[:, 0]
                        for score in scores:
                            ranking[current_id].score = score.item()
                            current_id += 1

            elif self.config.model_type == 'gemma':

                prompt_package = prompt_generator_t5(self.prompt_type, self.original_prompt_number, self.instruction,
                                                     self.output, self.tone, self.order, self.position, self.role)

                prompt_tem = prompt_package[1]  # pass prompt content

                conversation = [{"role": "user", "content": prompt_tem}]

                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                # <bos> has shown twice if decode batch_inputs

                data = [prompt.format(text=doc.text, query=query) for doc in ranking]

                if not self.has_printed_input_text:  # print a prompt for checking with configuration
                    print("prompt is \n" + data[0] + '\n')
                    # Set the flag to True after printing, to prevent future prints
                    self.has_printed_input_text = True

                print("prompt is \n" + data[0] + '\n')

                dataset = Text2TextGenerationDataset(data, self.tokenizer)

                if prompt_package[0] == "output_1":
                    yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_2":
                    yes_id = self.tokenizer.encode("True", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("False", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_3":
                    yes_id = self.tokenizer.encode("Highly", add_special_tokens=False)[0]
                    no_id = self.tokenizer.encode("Not", add_special_tokens=False)[0]
                elif prompt_package[0] == "output_4":
                    yes_id = 235310
                    no_id = 235276
                else:
                    raise ValueError(f"Invalid type of output {prompt_package[0]}")

                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    collate_fn=DataCollatorWithPadding(
                        self.tokenizer,
                        # max_length=512,
                        max_length=2048,
                        padding='longest',
                    ),
                    shuffle=False,
                    drop_last=False,
                    num_workers=0
                )

                current_id = 0
                with torch.no_grad():
                    for batch_inputs in tqdm(loader):
                        # from IPython import embed; embed(); exit()
                        batch_inputs = batch_inputs.to(self.llm.device)
                        logits = self.llm(input_ids=batch_inputs['input_ids'],
                                          attention_mask=batch_inputs['attention_mask'])
                        # print(logits)

                        logits = logits.logits[:, -1, :]
                        # print(logits)

                        yes_scores = logits[:, yes_id]
                        no_scores = logits[:, no_id]

                        batch_scores = torch.stack((yes_scores, no_scores), dim=1)
                        batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                        # probs = torch.nn.functional.softmax(logits, dim=-1)
                        # yes_scores = probs[:, yes_id].tolist()
                        # no_scores = probs[:, no_id].tolist()

                        scores = batch_scores[:, 0]
                        for score in scores:
                            ranking[current_id].score = score.item()
                            current_id += 1

            # labels1 = self.tokenizer.encode(f"<pad> Highly Relevant",
            #                                return_tensors="pt",
            #                                add_special_tokens=True).to(self.llm.device).repeat(self.batch_size, 1)
            #
            # labels2 = self.tokenizer.encode(f"<pad> Not Relevant",
            #                                return_tensors="pt",
            #                                add_special_tokens=True).to(self.llm.device).repeat(self.batch_size, 1)
            # current_id = 0
            # with torch.no_grad():
            #     for batch_inputs in tqdm(loader):
            #         self.total_compare += 1
            #         self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]
            #
            #         batch_labels1 = labels1 if labels1.shape[0] == len(batch_inputs['input_ids']) \
            #             else labels1[:len(batch_inputs['input_ids']), :]  # last batch might be smaller
            #         batch_labels2 = labels2 if labels2.shape[0] == len(batch_inputs['input_ids']) \
            #             else labels2[:len(batch_inputs['input_ids']), :]  # last batch might be smaller
            #         # self.total_prompt_tokens += batch_labels.shape[0] * batch_labels.shape[
            #         #     1]  # we count decoder inputs as part of prompt.
            #         batch_inputs = batch_inputs.to(self.llm.device)
            #         logits1 = self.llm(input_ids=batch_inputs['input_ids'],
            #                           attention_mask=batch_inputs['attention_mask'],
            #                           labels=batch_labels1).logits
            #         logits2 = self.llm(input_ids=batch_inputs['input_ids'],
            #                           attention_mask=batch_inputs['attention_mask'],
            #                           labels=batch_labels2).logits
            #         loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            #         scores1 = loss_fct(logits1.view(-1, logits1.size(-1)), batch_labels1.view(-1))
            #         scores2 = loss_fct(logits2.view(-1, logits2.size(-1)), batch_labels2.view(-1))
            #         scores1 = -1 * scores1.view(-1, batch_labels1.size(-1)).sum(dim=1)  # neg log prob
            #         scores2 = -1 * scores2.view(-1, batch_labels2.size(-1)).sum(dim=1)  # neg log prob
            #         batch_scores = torch.cat((scores1.unsqueeze(1), scores2.unsqueeze(1)), dim=1)
            #         batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
            #         scores = batch_scores[:, 0]
            #         for score in scores:
            #             ranking[current_id].score = score.item()
            #             current_id += 1

        ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return ranking

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
