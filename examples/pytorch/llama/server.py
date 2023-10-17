import socket
from numpysocket import NumpySocket
# 소켓 서버 설정
'''server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("127.0.0.1", 8889))
server_socket.listen(1)
print("Process A is waiting for connections...")
# 소켓 연결 대기
while True:
	client_socket, address = server_socket.accept()
	print(f"Connection established with {address}")

	# 결과 전송 후 소켓 종료
	final_reqs = client_socket.recv(1024)
	print("{} message".format(final_reqs.decode()))
	#client_socket.close()	
	print('hihi')
server_socket.close()'''

with NumpySocket() as s:
	s.bind(("", 9000))
	s.listen()
	conn, addr = s.accept()
	with conn:
		final_reqs = conn.recv()
		print(final_reqs.shape)