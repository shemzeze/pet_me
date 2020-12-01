import socket

s = socket.socket()
host = socket.gethostname()
ip_address = "192.168.0.10"
BUFFER_SIZE = 1024
# print(str(host))
port = 8080
s.bind((host, port))
s.listen(1)

while True:
    print("waiting for someone to connect...")

    conn, addr = s.accept()
    print(addr, "has connected to me")
    client_ip, client_port = addr
    # print(conn)
    # print(client_port)
    # s.connect()
    with open('received_file', 'wb') as f:
        # print ('file opened')
        while True:
            print('receiving data...')
            data = conn.recv(BUFFER_SIZE)
            print('data=%s', data)
            if not data:
                f.close()
                print('file close()')
                break
            # write data to a file
            f.write(data)
    f.close()
    print('Successfully get the file')
    s.close()
    print('connection closed')
