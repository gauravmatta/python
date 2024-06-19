import pika
from pika.exchange_type import ExchangeType


def alt_queue_message_processor(ch, method, properties, body):
    print(f" Alternate queue received new message: {body}")


def main_queue_message_processor(ch, method, properties, body):
    print(f" Main queue received new message: {body}")


connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()

channel.exchange_declare(exchange='alt_exchange', exchange_type=ExchangeType.fanout)
channel.exchange_declare(exchange='main_exchange', exchange_type=ExchangeType.direct,
                         arguments={'alternate-exchange': 'alt_exchange'})

channel.queue_declare(queue='alt_exchange_queue')
channel.queue_bind('alt_exchange_queue', 'alt_exchange')
channel.basic_consume(queue='alt_exchange_queue', auto_ack=True, on_message_callback=alt_queue_message_processor)

channel.queue_declare(queue='main_exchange_queue')
channel.queue_bind('main_exchange_queue', 'main_exchange', 'test')
channel.basic_consume(queue='main_exchange_queue', auto_ack=True, on_message_callback=main_queue_message_processor)

print("Started Consuming")
channel.start_consuming()
