from parse_rest.connection import register
from parse_rest.datatypes import Object
import key

register(key.APP_ID, key.REST_API_KEY)

class KeymochiTest(Object):
    pass

keymochi = KeymochiTest(user="Jean", mood="happy")
keymochi.save()
