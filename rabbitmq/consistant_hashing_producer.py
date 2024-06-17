import pika
from pika.exchange_type import ExchangeType

connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
channel.exchange_declare('hash_exchange', 'x-consistent-hash')
message = "I was routed via Hashing Exchange"
routing_key = "Hash me!"
channel.basic_publish(exchange='hash_exchange', routing_key=routing_key,
                      body=message)
print(f"sent message: {message}")
connection.close()
