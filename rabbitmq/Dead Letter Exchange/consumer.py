import pika
from pika.exchange_type import ExchangeType


def dlx_queue_message_processor(ch, method, properties, body):
    print(f" Dlx queue received new message: {body}")


def main_queue_message_processor(ch, method, properties, body):
    print(f" Main queue received new message: {body}")


connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()

channel.exchange_declare(exchange='dl_main_exchange', exchange_type=ExchangeType.direct)
channel.exchange_declare(exchange='dlx', exchange_type=ExchangeType.fanout)

channel.queue_declare(queue='dl_main_exchange_queue', arguments={
    'x-dead-letter-exchange': 'dlx',
    'x-message-ttl': 1000
})
channel.queue_bind('dl_main_exchange_queue', 'dl_main_exchange', 'test')
# channel.basic_consume(queue='dl_main_exchange_queue', auto_ack=True, on_message_callback=main_queue_message_processor)

channel.queue_declare(queue='dlx_queue')
channel.queue_bind('dlx_queue', 'dlx')
channel.basic_consume(queue='dlx_queue', auto_ack=True, on_message_callback=dlx_queue_message_processor)

print("Started Consuming")
channel.start_consuming()
