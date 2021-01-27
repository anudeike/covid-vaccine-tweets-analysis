
import asyncio
import time
import botometer
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

load_dotenv()
#
#
# async def part1(n: int) -> str:
#     i = 2
#     print(f"part1({n}) sleeping for {i} seconds.")
#     await asyncio.sleep(i) # sleeps for some time
#
#     result = f"result {n}-1"
#
#     print(f"\nReturning part1({n}) == {result}.")
#     return result
#
# async def part2(n: int, arg: str) -> str:
#     i = 2
#     print(f"part2{n, arg} sleeping for {i} seconds.")
#     await asyncio.sleep(i)
#     result = f"result{n}-2 derived from {arg}"
#     print(f"Returning part2{n, arg} == {result}.")
#     return result
#
# async def chain(n: int) -> None:
#     start = time.perf_counter()
#
#     # pass n in to part1
#     p1 = await part1(n)
#
#     # part 2 depends on part 1
#     p2 = await part2(n, p1)
#     end = time.perf_counter() - start
#     print(f"-->Chained result{n} => {p2} (took {end:0.2f} seconds).")
#
#
# async def aux_func(name, bom):
#     start = time.perf_counter()
#
#     r = await bom.check_account(name)
#
#     print(r["cap"])
#
#
#
# async def main(*args, bom):
#
#     # for each value in args
#     await asyncio.gather(*(aux_func(n, bom) for n in args))

async def main(**kwargs):
    pass
if __name__ == "__main__":
    # get info from botometer
    twitter_app_auth = {
        'consumer_key': os.getenv('TWITTER_API_KEY'),
        'consumer_secret': os.getenv('TWITTER_API_SECRET'),
    }

    rapidapi_key = os.getenv('RAPID_FIRE_KEY')

    bom = botometer.Botometer(wait_on_ratelimit=True,
                              rapidapi_key=rapidapi_key,
                              **twitter_app_auth)

    args = ["humanmgn", "traversymedia", "lee_sunkist"]
    start = time.perf_counter()

    # run the async function
    asyncio.run(main(*args, bom=bom))

    start = time.perf_counter()
    end = time.perf_counter() - start
    print(f"Program finished in {end:0.2f} seconds.")