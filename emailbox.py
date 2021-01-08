import argparse
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import Header
import json


class EmailBot(object):
    def __init__(self, settings_json_path):
        with open(settings_json_path) as f:
            self.settings = json.load(f)

    def connect(self):
        self.smtp = smtplib.SMTP_SSL(self.settings["smtp"], self.settings["port"])
        # self.smtp.connect(self.settings["smtp"], self.settings["port"])
        self.smtp.login(self.settings["sender"], self.settings["passport"])

    def quit(self):
        self.smtp.quit()

    def sendOne(self, data_dict={}):
        for key in ["title", "content", "From", "To", "Cc", "attachment"]:
            if key not in data_dict.keys():
                data_dict[key] = self.settings[key]
        msg = MIMEMultipart("auto email")
        msg.attach(MIMEText(data_dict["content"], _subtype="html", _charset="utf-8"))
        msg["Subject"] = Header(data_dict["title"], "utf-8")
        msg["From"] = data_dict["From"]
        # 添加发送人邮件信息（支持群发）
        msg["To"] = data_dict["To"]
        # 添加抄送人邮件信息
        msg["Cc"] = ";".join(data_dict["Cc"])  # 抄送邮箱
        # 添加附件信息(可包含多个附件)
        for file_path in data_dict["attachment"]:
            name = os.path.basename(file_path)  # 获取附件文件名
            file = MIMEApplication(open(file_path, "rb").read())
            file.add_header(
                "Content-Disposition", "attachment", filename=("utf-8", "", str(name))
            )  # 可以显示附件的中文名字
            msg.attach(file)
        self.connect()
        self.smtp.sendmail(msg["From"], msg["To"], msg.as_string())
        self.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-title", type=str, default="auto_email")
    parser.add_argument("-content", type=str, default="server process done!")
    parser.add_argument("-to", type=str, default="@qq.com")
    parser.add_argument("-smtp", type=str, default="smtp.qq.com")
    parser.add_argument("-sender", type=str, default="@qq.com")
    parser.add_argument("-passport", type=str, default="")
    parser.add_argument("-attachment", type=str, default="")
    parser.add_argument("-port", type=int, default=25)
    parser.add_argument("-Cc", type=str, default="")

    args = parser.parse_args()
    args.attachment = args.attachment.split(",") if args.attachment != "" else False
    args.to = args.to.split(",")
    args.Cc = args.Cc.split(",")

    smtp = smtplib.SMTP()
    smtp.connect(args.smtp, args.port)
    smtp.login(args.sender, args.passport)

    msg = MIMEMultipart("auto_email")
    msg.attach(MIMEText(args.content, _subtype="html", _charset="utf-8"))
    msg["Subject"] = Header(args.title, "utf-8")
    msg["From"] = args.sender
    # 添加发送人邮件信息（支持群发）
    msg["To"] = ";".join(args.to)
    # 添加抄送人邮件信息
    msg["Cc"] = ";".join(args.to)  # 抄送邮箱
    # 添加附件信息(可包含多个附件)
    if args.attachment:
        for a_path in args.attachment:
            name = os.path.basename(a_path)  # 获取附件文件名
            part = MIMEApplication(open(a_path, "rb").read())
            part.add_header(
                "Content-Disposition", "attachment", filename=("utf-8", "", str(name))
            )  # 可以显示附件的中文名字
            msg.attach(part)
    smtp.sendmail(msg["From"], msg["To"], msg.as_string())

    smtp.quit()
