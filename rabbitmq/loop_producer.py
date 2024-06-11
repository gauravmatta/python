import random

import pika
import time

connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
channel.queue_declare(queue='letterbox')
messageId = 1

while True:
    loop_message = f"This is a message from loop with id: {messageId}"
    channel.basic_publish(exchange='', routing_key='letterbox', body=loop_message)
    print(f"sent message :{loop_message}")
    time.sleep(random.randint(1, 4))
    messageId += 1
