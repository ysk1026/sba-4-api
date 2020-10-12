from com_sba_api.food.food_api import FoodApi
def initialize_routes(api):
    api.add_resource(FoodApi, '/api/foods')