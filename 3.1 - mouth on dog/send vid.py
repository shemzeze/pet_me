import socket

s = socket.socket()
host = socket.gethostname()
ip_address = "192.168.0.10"
BUFFER_SIZE = 1024
# print(str(host))
port = 8080
s.bind((host, port))

while True:
    s.listen(1)
    print("waiting for someone to connect...")
    conn, addr = s.accept()
    print('Got connection from', addr)
    # data = conn.recv(1024)
    # print('Server received', repr(data))

    # f = open(filename, 'rb')
    with open('new.mp4', 'rb') as f:
        lll = f.read(1024)
        while lll:
            data = conn.send(lll)
            print('Sent ', repr(lll))
            print(len(lll))
            lll = f.read(1024)
            # if not data:
            #     break
    f.close()
    print('Done sending')
    conn.close()
