from flask import Flask, request
from flask_restful import Resource, Api, reqparse, abort
import json
from parse_rest.connection import register
from parse_rest.datatypes import Object
import key

register(key.APP_ID, key.REST_API_KEY)

class DataChunk(Object):
    pass

#keymochi = KeymochiTest(user="Jean", mood="happy")
#keymochi.save()

print(DataChunk.Query.all())

