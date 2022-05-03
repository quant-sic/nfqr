import json


def geometry(string):
    return [int(obj) for obj in string.split("x")]


def json_file(path):
    if path is not None:
        with open(path, "r") as buff:
            return json.load(buff)
    else:
        return {}


def csint(string):
    return [int(obj) for obj in string.split(",") if obj != ""]


def csstr(string):
    if len(string):
        return string.split(",")
    else:
        return []
