"""Read GGUF metadata/tensor descriptors without loading tensor payloads."""
import struct


def read_header(path):
    with open(path, "rb") as file:
        def take(fmt):
            return struct.unpack("<" + fmt, file.read(struct.calcsize("<" + fmt)))[0]

        def string():
            return file.read(take("Q")).decode()

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, count = take("I"), take("Q")
                return [value(element) for _ in range(count)]
            return take({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                         10: "Q", 11: "q", 12: "d"}[kind])

        assert file.read(4) == b"GGUF" and take("I") == 3
        tensor_count, metadata_count = take("Q"), take("Q")
        metadata = {}
        for _ in range(metadata_count):
            name, kind = string(), take("I")
            metadata[name] = value(kind)
        tensors = {}
        for _ in range(tensor_count):
            name, dims = string(), take("I")
            shape = [take("Q") for _ in range(dims)]
            tensors[name] = dict(shape=shape, type=take("I"), offset=take("Q"))
        alignment = metadata.get("general.alignment", 32)
        offset = (file.tell() + alignment - 1) // alignment * alignment
        return metadata, tensors, offset
