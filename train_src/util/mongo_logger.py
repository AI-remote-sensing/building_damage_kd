from pymongo import MongoClient
import time
import pandas as pd


class Logger(object):
    def __init__(self, db, collection):
        super(Logger, self).__init__()
        self.CONN = MongoClient("localhost")
        self.DB = self.CONN[db]
        self.COLLECTION = self.DB[collection]
        self.data_dict = {"default": {}}
        self.log_id = int(time.time())

    def add_attr(self, key, value, name="default"):
        if name not in self.data_dict.keys():
            self.data_dict[name] = {}
        self.data_dict[name][key] = value

    def insert_into_db(self, name="default", shared=False):
        self.data_dict[name]["time"] = time.time()
        self.data_dict[name]["name"] = name
        self.data_dict[name]["log_id"] = self.log_id
        self.COLLECTION.insert_one(self.data_dict[name])
        if not shared:
            self.data_dict[name] = {}

    def save_df(self, save_path, name="default"):
        format_ = save_path.split(".")[-1]
        if format_ not in ["csv", "xls"]:
            print("need to be csv or xls format")
        else:
            data_list = self.COLLECTION.find({"log_id": self.log_id, "name": name})
            df = pd.DataFrame(data_list)
            if format_ == "csv":
                df.to_csv(save_path)
            elif format_ == "xls":
                df.to_excel(save_path)


if __name__ == "__main__":
    logger = Logger("test", "logger")
    for i in range(100):
        if i % 5 == 0:
            logger.add_attr("batch", i, "per5")
            logger.add_attr("per5", i)
            logger.insert_into_db("per5")
        logger.add_attr("batch", i)
        logger.add_attr("content", "same text")
        logger.insert_into_db()
    logger.save_df("test.csv")
    logger.save_df("per5.xls")
