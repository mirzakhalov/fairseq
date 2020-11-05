with open("destdir/valid.multi-en.en.bin", "rb") as f:
    byte = f.read(1)
    while byte:
        # Do stuff with byte.
        byte = f.read(1)
        print(byte)