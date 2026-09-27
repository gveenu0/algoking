# active_market_order = [
#  {"order_id": "101", "user_id": "user_A", "asset": "BTC", "type": "BUY", "price": 65000},
#  {"order_id": "102", "user_id": "user_A", "asset": "BTC", "type": "SELL", "price": 65100},
#  {"order_id": "103", "user_id": "user_B", "asset": "ETH", "type": "BUY", "price": 3500},
#  {"order_id": "104", "user_id": "user_A", "asset": "ETH", "type": "BUY", "price": 3490},
#  {"order_id": "105", "user_id": "user_C", "asset": "BTC", "type": "SELL", "price": 65200},
#  {"order_id": "106", "user_id": "user_B", "asset": "ETH", "type": "SELL", "price": 3510}
# ]
#
# users_active_buy = {}
# users_active_sell ={}
# res = []
#
#
# for order in active_market_order:
#     if order['type'] == "BUY":
#         if order['user_id'] not in users_active_buy:
#             users_active_buy[order['user_id']] = {order['asset']:order['order_id']}
#         else:
#             users_active_buy[order['user_id']][order['asset']] = order['order_id']
#     elif order['type'] == "SELL":
#         if order['user_id'] not in users_active_sell:
#             users_active_sell[order['user_id']] = {order['asset']:order['order_id']}
#         else:
#             users_active_sell[order['user_id']][order['asset']] = order['order_id']
#     else:
#         raise("Unknown order type")
#
# for user in users_active_buy:
#     if user in users_active_sell:
#         for assets in users_active_buy[user]:
#             if assets in users_active_sell[user]:
#                 res.append({'user_id': user, 'assets': assets, 'buy_orders': [users_active_buy[user][assets]], 'sell_orders': [users_active_sell[user][assets]]})
#
#
# print(res)
#
#
#
#


l = [1,2,3,4,5,6,7,8]
print(l[3::-1])
del l[2]
print(l)
