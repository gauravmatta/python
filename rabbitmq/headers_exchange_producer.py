import pika
from pika.exchange_type import ExchangeType

connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
channel.exchange_declare('headers_exchange', exchange_type=ExchangeType.headers)
message = "This message will be sent with headers"
channel.queue_declare(queue='letterbox')
channel.basic_publish(exchange='headers_exchange', routing_key='',
                      properties=pika.BasicProperties(headers={'name': 'gaurav'}),
                      body=message)
print(f"sent message: {message}")
connection.close()
