import pika
from pika.exchange_type import ExchangeType

connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
channel.exchange_declare(exchange='mytopicexchange', exchange_type=ExchangeType.topic)
message = "This message is to be routed to topics based on "
channel.basic_publish(exchange='mytopicexchange', routing_key='user.asia.payments', body=message + "user.asia.payments")
channel.basic_publish(exchange='mytopicexchange', routing_key='suppliers.india.payments',
                      body=message + "suppliers.india.payments")
channel.basic_publish(exchange='mytopicexchange', routing_key='suppliers.asia.payments',
                      body=message + "suppliers.asia.payments")
print(f"sent message: {message}")
connection.close()
