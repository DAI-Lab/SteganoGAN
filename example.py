from deepsteganography import Steganographer
s = Steganographer()
s.encode("images/flag.jpg", "Hello!", "images/flag.out.png")
print(s.decode("images/flag.out.png"))
