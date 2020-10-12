from flask_restful import Resource
from flask import Response, jsonify
from com_sba_api.food.food_dao import FoodDao

class FoodApi(Resource):
    def __init__(self):
        self.dao = FoodDao()
        
    def get(self):
        food = self.dao.select_foods()
        return jsonify(food[0])
