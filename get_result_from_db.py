from pymongo import MongoClient

db = "building_damage_kd"
collection = "v0_loc"
CONN = MongoClient("localhost")
DB = CONN[db]
COLLECTION = DB[collection]

find_list = COLLECTION.find({"name": "time_difference"})
find_set = set([x["log_id"] for x in list(find_list)])
print(find_set)
# 1609952015, 1609941644, 1609959453, 1609945583

for log_id in find_set:
    print(log_id)
    print(COLLECTION.find_one({"name": "info", "log_id": log_id}))

    score_list = [
        dict_["score"]
        for dict_ in list(COLLECTION.find({"name": "default", "log_id": log_id}))
    ]
    print(score_list, max(score_list))
    print(
        COLLECTION.find({"name": "default", "log_id": log_id})[
            score_list.index(max(score_list))
        ]
    )
