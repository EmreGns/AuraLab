import socket

# Server IP ve Port (ESP'nin gönderdiği yere göre)
HOST = '0.0.0.0'  # Tüm IP'lerden dinler
PORT = 9090       # ESP kodunda yazdığımız port

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"Listening on port {PORT}...")

while True:
    client_socket, addr = server.accept()
    print(f"Connection from {addr}")

    data = client_socket.recv(1024)
    print(f"Received: {data.decode()}")

    client_socket.close()
