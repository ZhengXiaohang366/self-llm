import concurrent.futures
import time

def task1():
    for i in range(5):
        print(f"Task 1: {i}")
        time.sleep(1)
    return "Task 1 finished"

def task2():
    for i in range(5):
        print(f"Task 2: {i}")
        time.sleep(1)
    return "Task 2 finished"

with concurrent.futures.ProcessPoolExecutor() as executor:
    # 提交任务
    future1 = executor.submit(task1)
    future2 = executor.submit(task2)

    # 获取任务结果
    result1 = future1.result()
    result2 = future2.result()

print(result1)
print(result2)