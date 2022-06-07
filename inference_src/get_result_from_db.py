#%%
from pymongo import MongoClient

db = "building_damage_kd"
collection = "v1_cls"
CONN = MongoClient("localhost")
DB = CONN[db]
COLLECTION = DB[collection]

#%%
find_list = COLLECTION.find({"name": "time_difference"})
find_set = set([x["log_id"] for x in list(find_list)])
print(find_set)
# 1610386176, 1610355962, 1610352055
#%%
for log_id in find_set:
    print(log_id)
    print(COLLECTION.find_one({"name": "info", "log_id": log_id}))

    # score_list = [
    #     dict_["score"]
    #     for dict_ in list(COLLECTION.find({"name": "default", "log_id": log_id}))
    # ]
    # print(score_list, max(score_list))
    # print(
    #     COLLECTION.find({"name": "default", "log_id": log_id})[
    #         score_list.index(max(score_list))
    #     ]
    # )

# %%
