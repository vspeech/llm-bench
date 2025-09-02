# coding: utf-8
import json
import sys
import asyncio
import aiohttp
import time,re

# 并发请求数量
CONCURRENCY_LIMIT = 1  # 可以根据需要调整

# 根据业务进行更改
HOST_IP="60.###.###.###"
USER_HEADER_HOST_NAME="url.domain-name.com"

URL=f"http://{HOST_IP}/infer"
total_run_time=0
run_count=0
error_count=0

async def fetch(session, url, data):
    start_time = time.time()
    try:
        user_headers={'Content-Type':'application/json'}
        user_headers["Host"]=USER_HEADER_HOST_NAME
        
        global total_run_time,run_count,error_count
        async with session.post(url,headers=user_headers, json=data) as response:
            resp = await response.json()
            time_used=time.time()-start_time
            total_run_time+=time_used
            run_count+=1
            return resp
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        error_count+=1
        return None

async def process_line(session, semaphore, line):
    async with semaphore:  # 限制并发数量
      try:
            
            entities_line = line.strip()
            
            # data 发给服务器的内容
            # 需要根据业务自行组织
            data = {}
            # 这里读取一行，并将数据写入到data中
            data["entities"]=entities_line.split("-")
            data["order"]=-1
            
            url = URL
            resp = await fetch(session, url, data)
            if resp:
                print(len(resp), flush=True)
        except Exception as e:
            sys.stderr.write(f"{line.strip()}\t{str(e)}\n")

async def main():
    start_time = time.time()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)  # 创建信号量
    async with aiohttp.ClientSession() as session:
        tasks = []
        for line in sys.stdin:
            task = asyncio.create_task(process_line(session, semaphore, line))
            tasks.append(task)
        await asyncio.gather(*tasks)  # 并发执行所有任务
    end_time = time.time()
    duration = end_time - start_time
    global total_run_time,run_count,error_count
    average_rt=total_run_time/run_count
    sys.stderr.write(f"请求数量: {run_count}\n")
    sys.stderr.write(f"并发数量: {CONCURRENCY_LIMIT}\n")
    sys.stderr.write(f"执行时长: {duration}秒\n")
    sys.stderr.write(f"平均执行时长: {average_rt}秒\n")
    sys.stderr.write(f"错误数量: {error_count}\n")
  
if __name__ == "__main__":
    # 启动命令示例： 从 test_queries 读取50行，并进行压测
    # 建议tes_queries 多放点数据
    #  head -n 50 test_queries | python chunmei_graph.py
    
    # 扩充数据
    #     awk 'NR<=2 {lines[NR]=$0} END{for(i=1;i<=500;i++) print lines[1] "\n" lines[2]}' test_queries | python chunmei_graph.py

    asyncio.run(main())
