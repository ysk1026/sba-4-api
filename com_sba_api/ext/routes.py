from com_sba_api.food.food_router import FoodRouter

def initialize_routes(api):
    api.add_resource(FoodRouter, '/api/foods')