# STRUCT - TUTORIALS

This document teaches you how to use Python's `struct` module to store and retrieve binary data.

# 1. Importing Struct

First, import the module.

```python
import struct
```

# 2. Packing Data

Packing converts Python values into binary data.

For example:

```python
import struct

number = 3.14

data = struct.pack("f", number)

print(data)
```

The `"f"` tells Python to store the number as a 32-bit floating-point value.

# 3. Unpacking Data

Unpacking converts binary data back into Python values.

```python
import struct

data = struct.pack("f", 3.14)

number = struct.unpack("f", data)

print(number)
```

Output:

```text
(3.140000104904175,)
```

Notice that `unpack()` returns a tuple.

# 4. Format Characters

Struct uses format characters to determine how values are stored.

Some important characters:

| Character | Data Type        | Size    |
| --------- | ---------------- | ------- |
| `f`       | Float            | 4 bytes |
| `d`       | Double           | 8 bytes |
| `i`       | Integer          | 4 bytes |
| `h`       | Short            | 2 bytes |
| `b`       | Signed byte      | 1 byte  |
| `?`       | Boolean          | 1 byte  |
| `I`       | Unsigned integer | 4 bytes |
| `B`       | Unsigned byte    | 1 byte  |

# 5. Packing Multiple Numbers

You can pack multiple values into one binary object.

```python
import struct

data = struct.pack("fff", 1.5, 2.5, 3.5)

print(data)
```

The `"fff"` means three 32-bit floating-point numbers.

# 6. Unpacking Multiple Numbers

You can retrieve multiple values using the same format.

```python
import struct

data = struct.pack("fff", 1.5, 2.5, 3.5)

numbers = struct.unpack("fff", data)

print(numbers)
```

Output:

```text
(1.5, 2.5, 3.5)
```

# 7. Writing Binary Files

You can save packed data directly into files.

```python
import struct

numbers = [0.5, 0.8, 0.2, 0.9]

with open("numbers.bin", "wb") as file:
    file.write(struct.pack("<4f", *numbers))
```

Here:

* `<` means little-endian byte order.
* `4` means four values.
* `f` means 32-bit floating-point numbers.
* `*numbers` passes the list elements as separate arguments.

# 8. Reading Binary Files

You can load the previously saved values.

```python
import struct

with open("numbers.bin", "rb") as file:
    data = file.read()

numbers = struct.unpack("<4f", data)

print(numbers)
```

Output:

```text
(0.5, 0.800000011920929, 0.20000000298023224, 0.8999999761581421)
```

# 9. Appending Binary Data

You can append more binary data without overwriting existing contents.

```python
import struct

new_numbers = [0.3, 0.7]

with open("numbers.bin", "ab") as file:
    file.write(struct.pack("<2f", *new_numbers))
```

`ab` means append binary.

# 10. Calculating Storage Size

You can calculate how many bytes a format requires using `struct.calcsize()`.

```python
import struct

size = struct.calcsize("<4f")

print(size)
```

Output:

```text
16
```

Four floating-point numbers take 16 bytes because each one takes 4 bytes.

# 11. Storing Integers

You can also store integers using struct.

```python
import struct

number = 250

data = struct.pack("<i", number)

with open("integer.bin", "wb") as file:
    file.write(data)
```

To retrieve it:

```python
import struct

with open("integer.bin", "rb") as file:
    data = file.read()

number = struct.unpack("<i", data)[0]

print(number)
```

Output:

```text
250
```

# 12. Storing Strings

Struct doesn't directly support variable-length strings, but you can store strings as bytes.

```python
import struct

text = "Hello World!"

data = text.encode("utf-8")

with open("text.bin", "wb") as file:
    file.write(data)
```

To retrieve it:

```python
with open("text.bin", "rb") as file:
    data = file.read()

text = data.decode("utf-8")

print(text)
```

Output:

```text
Hello World!
```

Note: This example uses Python's string encoding and decoding rather than struct packing.

# 13. Storing Multiple Data Types

You can combine different data types into one binary structure.

```python
import struct

name = b"Logan"
age = 13
height = 1.65

data = struct.pack("<5sif", name, age, height)

with open("person.bin", "wb") as file:
    file.write(data)
```

Here:

* `5s` means a 5-byte string.
* `i` means a 32-bit integer.
* `f` means a 32-bit float.

To retrieve the information:

```python
import struct

with open("person.bin", "rb") as file:
    data = file.read()

name, age, height = struct.unpack("<5sif", data)

print(name.decode("utf-8"))
print(age)
print(height)
```

Output:

```text
Logan
13
1.649999976158142
```

# 14. Using Struct Without Files

You don't always need to write binary data to a file.

You can pack and unpack data directly in memory.

```python
import struct

data = struct.pack("<ii", 100, 200)

first, second = struct.unpack("<ii", data)

print(first)
print(second)
```

Output:

```text
100
200
```

# 15. Reading Binary Data in Chunks

You can read a binary file in smaller portions.

```python
import struct

with open("numbers.bin", "rb") as file:
    while True:
        data = file.read(4)

        if not data:
            break

        number = struct.unpack("<f", data)[0]

        print(number)
```

Each read retrieves one 32-bit floating-point number.

# 16. Using struct.iter_unpack()

`iter_unpack()` lets you unpack repeated structures without manually creating a loop for each individual value.

```python
import struct

numbers = [1.5, 2.5, 3.5, 4.5]

data = struct.pack("<4f", *numbers)

for number in struct.iter_unpack("<f", data):
    print(number[0])
```

Output:

```text
1.5
2.5
3.5
4.5
```

# 17. Using struct.pack_into()

`pack_into()` allows you to pack values directly into an existing mutable buffer.

```python
import struct

buffer = bytearray(8)

struct.pack_into("<i", buffer, 0, 100)
struct.pack_into("<i", buffer, 4, 200)

print(buffer)
```

Here:

* `buffer` is the destination.
* `0` and `4` are the byte offsets.
* `100` and `200` are the values being packed.

# 18. Using struct.unpack_from()

`unpack_from()` retrieves values from an existing buffer without requiring you to extract the bytes manually.

```python
import struct

data = struct.pack("<ii", 100, 200)

first = struct.unpack_from("<i", data, 0)[0]
second = struct.unpack_from("<i", data, 4)[0]

print(first)
print(second)
```

Output:

```text
100
200
```

# 19. Checking Binary File Integrity

You can check whether a binary file contains the expected number of bytes.

```python
import struct

numbers = [1.0, 2.0, 3.0]

with open("numbers.bin", "wb") as file:
    file.write(struct.pack("<3f", *numbers))

with open("numbers.bin", "rb") as file:
    data = file.read()

expected_size = struct.calcsize("<3f")

if len(data) == expected_size:
    print("Binary file has the expected size!")
else:
    print("Binary file has an incorrect size!")
```

This checks the file size, but doesn't guarantee that its contents are correct.

# 20. Using struct.Struct()

You can create a reusable Struct object when repeatedly using the same format.

```python
import struct

number_format = struct.Struct("<if")

data = number_format.pack(100, 3.14)

number, decimal = number_format.unpack(data)

print(number)
print(decimal)
```

This avoids repeatedly parsing the same format string.

# 21. What's Next?

After learning the basics, explore:

* Binary file formats.
* Little-endian and big-endian.
* Packing and unpacking arrays.
* Binary headers.
* Byte buffers.
* `struct.pack_into()`.
* `struct.unpack_from()`.
* `struct.iter_unpack()`.
* `struct.Struct()`.

# 22. Important Notes

* `struct` converts Python values into binary representations.
* Packing and unpacking formats must match.
* `struct` uses fixed-size data types.
* Binary files don't automatically store information about their contents.
* `unpack()` returns a tuple.
* `pack()` returns a bytes object.
* Little-endian and big-endian determine byte order.
* Always ensure that the binary data has the correct size before unpacking.

---

* This content was generated by ChatGPT. Be sure to check sources or correct stuff inside the doc before using it.
