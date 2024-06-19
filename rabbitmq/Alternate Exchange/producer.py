import pika
from pika.exchange_type import ExchangeType

connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()

channel.exchange_declare(exchange='alt_exchange', exchange_type=ExchangeType.fanout)
channel.exchange_declare(exchange='main_exchange', exchange_type=ExchangeType.direct,
                         arguments={'alternate-exchange': 'alt_exchange'})
message = "Hello this is my first message"
channel.basic_publish(exchange='main_exchange', routing_key='test', body=message)
channel.basic_publish(exchange='main_exchange', routing_key='dev', body=message)
print(f"sent message: {message}")
connection.close()
