import os
import asyncio
import time
from datetime import datetime
import pandas as pd
import aiohttp
from collections import defaultdict
import json
from typing import List
import random,traceback
from threading import Lock
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field 
from copy import deepcopy


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content:Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length",""]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str =""
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(
        default=None, description="data about request and response")



def load_prompts(file_path: str, repeat_time:int=100) -> List[dict]:
    """Load prompts from a file, one prompt per line"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Remove empty lines and strip whitespace
            prompts = json.load(f)
        if not prompts:
            raise ValueError("No valid prompts found in the file")
        
    except Exception as e:
        print(f"Error loading prompts: {e}")
        # Provide a default prompt in case of error
        prompts=[{"messages": [{"role": "user", "content": "Tell me a short joke about programming"}]}]
    
    prompts = repeat_time*prompts
    return prompts
    

class PromptSelector:
    def __init__(self, prompts: List[dict], selection_strategy: str = "random"):
        """
        Initialize prompt selector with a list of prompts
        selection_strategy: "random" or "sequential"
        """
        self.prompts = prompts
        self.strategy = selection_strategy
        self.current_index = 0
        self._lock = Lock()  # Add lock for thread safety
        
    def get_next_prompt(self) -> dict:
        """Get the next prompt based on the selection strategy in a thread-safe manner"""
        if self.strategy == "random":
            prompt=random.choice(self.prompts)
        elif self.strategy == "normal":
            with self._lock:
                if self.current_index <len(self.prompts):
                    prompt=self.prompts[self.current_index]
                    self.current_index+=1
                else:
                    prompt=None
        else:  # sequential
            with self._lock:
                prompt = self.prompts[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.prompts)
                
        return deepcopy(prompt)

class TokenUsageTracker:
    def __init__(self):
        self._lock = Lock()  # Add lock for thread safety
        # Store token usage per minute window
        self.prompt_tokens = defaultdict(float)
        self.completion_tokens = defaultdict(float)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_first_response_time = 0
        self.total_run_time=0
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()

    def add_usage(self, start_time: float, end_time: float, first_response_time: float, prompt_tokens: int, completion_tokens: int):
        """
        Calculate token distribution per minute based on request duration
        Thread-safe implementation using locks
        """
        duration = end_time - start_time
        print(f"{start_time} --  {end_time} 用时:{duration}")
        completion_tokens_per_sec = completion_tokens / duration
        
        start_minute = int(start_time / 60)
        end_minute = int(end_time / 60)
        
        with self._lock:
            self.total_run_time+=duration
            self.total_first_response_time += first_response_time      
            self.prompt_tokens[start_minute] += prompt_tokens
            
            # Handle single minute case
            if start_minute == end_minute:
                self.completion_tokens[start_minute] += completion_tokens
            else:
                # First minute - partial
                first_minute_end = (start_minute + 1) * 60
                first_minute_duration = first_minute_end - start_time
                self.completion_tokens[start_minute] += completion_tokens_per_sec * first_minute_duration
                
                # Last minute - partial
                last_minute_start = end_minute * 60
                last_minute_duration = end_time - last_minute_start
                self.completion_tokens[end_minute] += completion_tokens_per_sec * last_minute_duration
                
                # Full minutes in between
                if end_minute - start_minute > 1:
                    for minute in range(start_minute + 1, end_minute):
                        self.completion_tokens[minute] += completion_tokens_per_sec * 60
            
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.request_count += 1

    def increment_error_count(self):
        """Thread-safe method to increment error count"""
        with self._lock:
            self.error_count += 1

    def print_stats(self):
        """Print current statistics in a thread-safe manner"""
        
        current_time = time.time()
        elapsed_minutes = (current_time - self.start_time) / 60
        if self.request_count ==0:
            return
        with self._lock:
            total_tokens = self.total_prompt_tokens + self.total_completion_tokens
            overall_tpm = total_tokens / elapsed_minutes if elapsed_minutes > 0 else 0
            # Create a copy of the data to avoid holding the lock for too long
            prompt_tokens_copy = dict(self.prompt_tokens)
            completion_tokens_copy = dict(self.completion_tokens)
            average_first_response_time=self.total_first_response_time/self.request_count
            average_run_time= self.total_run_time/self.request_count
            
        print("\nToken Usage Statistics:")
        print(f"Total Requests: {self.request_count}")
        print(f"Total Errors: {self.error_count}")
        print(f"Total Prompt Tokens: {self.total_prompt_tokens:.0f}")
        print(f"Total Completion Tokens: {self.total_completion_tokens:.0f}")
        print(f"Average First Token: {average_first_response_time:.2f}")
        print(f"Average Run Time: {average_run_time:.2f}")
        print(f"Total Run Time: {self.total_run_time:.2f}")
        print(f"Overall TPM: {overall_tpm:.2f}")

        # Process the data outside the lock
        data = []
        for minute in sorted(set(prompt_tokens_copy.keys()) & set(completion_tokens_copy.keys())):
            prompt_tokens = prompt_tokens_copy[minute]
            completion_tokens = completion_tokens_copy[minute]
            total_tokens_minute = prompt_tokens + completion_tokens
            
            data.append({
                'Minute': minute,
                'Time': datetime.fromtimestamp(minute * 60).strftime('%H:%M:%S'),
                'Prompt Tokens': f"{prompt_tokens:.0f}",
                'Completion Tokens': f"{completion_tokens:.0f}",
                'Total Tokens': f"{total_tokens_minute:.0f}",
                'TPM': f"{total_tokens_minute:.2f}"
            })
        
        if data:
            df = pd.DataFrame(data)
            print("\nPer-minute Statistics:")
            print(df.to_string(index=False))

async def process_stream_response(response, start_time):
    """Process streaming response and extract the final usage information"""
    usage = None
    first_response_time = None
    try:
        async for line in response.content:
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    if decoded_line.strip() == 'data: [DONE]':
                        continue
                    
                    json_str = decoded_line[6:]  # Remove 'data: ' prefix
                    json_data = json.loads(json_str)
                    # print(json_data)
                    
                    chunk = ChatCompletionStreamResponse(**json_data)
                    if len(chunk.choices) != 0:
                        if chunk.choices[0].delta.content is not None or chunk.choices[0].delta.reasoning_content is not None:
                            if first_response_time is None:
                                first_response_time = round(time.time() - start_time, 2)
                    # The last chunk contains the usage information
                    if chunk.usage is not None:
                        # print(json_data)
                        usage = chunk.usage
                
    except Exception as e:
        print(f"Error processing stream: {e}")
        print(json_data)
        traceback.print_exc()
        return None, None
    
    return usage, first_response_time

async def continuous_requests(tracker: TokenUsageTracker, prompt_selector: PromptSelector, worker_id: int, end_time: float):
    """Continuously make requests until the end time is reached"""
    
    
    async with aiohttp.ClientSession() as session:
        while time.time() < end_time:
            try:
                prompt = prompt_selector.get_next_prompt()
                
                if prompt is None:
                    print(f"协程{worker_id}退出--")
                    break
                
                start_time = time.time()
                
                request_item = {key:value for key,value in prompt.items() if key in openai_params}
                
                global GLOBAL_COUNT
                GLOBAL_COUNT+=1
                print(f"GLOBAL_COUNT 值:{GLOBAL_COUNT}")
                
                request_item["messages"][0]["content"] = f"这是第{GLOBAL_COUNT}条请求，"+request_item["messages"][0]["content"]
                
                payload = {
                    "model":REQUEST_MODEL,
                    "user": "system-bench",
                    "stream": True,
                    **request_item
                }
                # payload["stream_options"]={"include_usage":True}
                
                async with session.post(OPENAI_API_URL, headers=HEADERS, json=payload) as response:
                    if response.status == 200:
                        usage, first_response_time = await process_stream_response(response, start_time)
                        response_end_time=time.time()
                        if usage:
                            tracker.add_usage(
                                start_time,
                                response_end_time,
                                first_response_time,
                                usage.prompt_tokens,
                                usage.completion_tokens
                            )
                        else:
                            print(f"Worker {worker_id}: Could not extract usage information")
                            tracker.increment_error_count()
                    else:
                        print(f"Worker {worker_id}: API returned status code {response.status}")
                        response_text = await response.text()
                        print(f"Response: {response_text}")
                        tracker.increment_error_count()
                
                # Add a small delay between requests to avoid overwhelming the API
                await asyncio.sleep(0.001)
                    
            except Exception as e:
                print(f"Worker {worker_id}: Error in request: {e}")
                traceback.print_exc()
                tracker.increment_error_count()
                # Add a delay before retrying after an error
                await asyncio.sleep(0.001)

async def print_stats_periodically(tracker: TokenUsageTracker, end_time: float):
    """Print statistics every minute"""
    while time.time() < end_time:
        await asyncio.sleep(60)  # Wait for 1 minute
        tracker.print_stats()

async def run_benchmark(duration_seconds: int, prompt_selector: PromptSelector, concurrent:int=2):
    """Run benchmark with two continuous request workers"""
    tracker = TokenUsageTracker()
    end_time = time.time() + duration_seconds
    
    # Create two workers
    workers = [
        continuous_requests(tracker, prompt_selector, i, end_time)
        for i in range(concurrent)
    ]
    
    # Create stats printer task but run it separately
    stats_printer = asyncio.create_task(print_stats_periodically(tracker, end_time))
    
    # Run workers concurrently
    await asyncio.gather(*workers)
    
    # Cancel the stats printer task
    stats_printer.cancel()
    try:
        await stats_printer
    except asyncio.CancelledError:
        pass
    
    return tracker

async def main():
    # Benchmark parameters
    duration_seconds = 300  # 3 minutes
    prompts_file = "prompts.json"  # Path to your prompts file
    selection_strategy = "random"  # or "sequential"
    # selection_strategy = "normal"  
    print(f"Loading prompts from: {prompts_file}")
    prompts = load_prompts(prompts_file,repeat_time=100)
    prompt_selector = PromptSelector(prompts, selection_strategy)
    
    print(f"Starting benchmark:")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Number of prompts loaded: {len(prompts)}")
    print(f"Selection strategy: {selection_strategy}")
    
    tracker = await run_benchmark(duration_seconds, prompt_selector,concurrent=900)
    
    print("\nFinal Statistics:")
    tracker.print_stats()

if __name__ == "__main__":

    OPENAI_API_URL = "https://url/api/v2/services/aigc/text-generation/v1/chat/completions"

    OPENAI_API_KEY = "correct key"

    # 模型请求
    REQUEST_MODEL="DeepSeek-R1"


    HEADERS = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    openai_params = [
            "messages", "frequency_penalty", "max_tokens", "presence_penalty",
            "temperature", "stop", "top_p", "top_k", "min_p", "repetition_penalty"
        ]

    GLOBAL_COUNT=300
    asyncio.run(main())
