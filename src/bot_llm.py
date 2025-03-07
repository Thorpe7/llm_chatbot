import os
import torch
import textstat
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface.llms import HuggingFacePipeline

from utils.output_schemas import ModelResponse

class LlamaInstruct:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        model_token = os.getenv("INSTRUCT_TOKEN")

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=model_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=model_token,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).to(self.device)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

        self.pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.2,
            torch_dtype=torch.float16,
            device_map="auto",
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        
        # answer_schema = ResponseSchema(name="answer", description="This is the answer to the user's question.")

        # response_schemas = [answer_schema]
        
        # self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        self.output_parser = PydanticOutputParser(pydantic_object=ModelResponse)
        self.format_instructions = self.output_parser.get_format_instructions()

        # Structured Prompt Template (forces JSON output)
        self.prompt = PromptTemplate(
            template=("""
                      You are a master software engineer who specializes in assisting clients with documentation and understand how to use your platform. 
                      Given the following question provided by an user: 
                      ```{question}```
                      Answer the question according to this information about your platform and provide a code example if requested by the user:
                      ```{context}```
                      Format the output as JSON with the following keys:
                      {format_instructions}
                      """
            ),
            input_variables=["question", "context"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions}
        )

        # Define the chain without the parser
        self.chain = self.prompt | self.llm | self.output_parser


    def generate(self, question: str, context: str) -> dict:
        """Runs the pipeline, and extracts the answer."""
        
        # self._update_pipeline(question)

        # Pass the formatted prompt directly to the LLM
        result = self.chain.invoke({"question": question, "context": context, "format_instructions": self.format_instructions})
        print(result)
        # result = self.output_parser.parse(result)
        
        return result

    def _determine_max_tokens(self, question:str) -> int:
        """ Use textstat to evaluate complexity of user query for max tokens."""
        complexity_score = textstat.flesch_reading_ease(question)

        if complexity_score > 60:
            return 100  # Simple
        elif complexity_score > 40:
            return 200  
        elif complexity_score > 20:
            return 300  
        else:
            return 500 # Complex
    
    def _update_pipeline(self,question) -> None:
        max_new_tokens = self._determine_max_tokens(question)
        self.pipeline = pipeline(
            "text-generation",
            model=self.pipeline.model,
            tokenizer=self.pipeline.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.2,
            torch_dtype=torch.float16,
            device_map="auto",
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            return_full_text=False
        )


if __name__ == "__main__":
    TestModel = LlamaInstruct()
    
    test_context = "Paris is the capital of France. France is a country in Europe."
    test_question = "What is the capital of France?"
    output = TestModel.generate(test_question, test_context)
    
    print("\n========= MODEL OUTPUT =========")
    print(print(type(output)))
    print(output)
    print("================================")
