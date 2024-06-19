import pika
from pika.exchange_type import ExchangeType

connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()

channel.exchange_declare(exchange='dl_main_exchange', exchange_type=ExchangeType.direct)

message = "This message will expire"
channel.basic_publish(exchange='dl_main_exchange', routing_key='test', body=message)
channel.basic_publish(exchange='dl_main_exchange', routing_key='rest', body=message)
print(f"sent message: {message}")
connection.close()
