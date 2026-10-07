from io import BytesIO
import unittest
import zipfile
from .remote_zip import RemoteZip


class MemoryZip(RemoteZip):
    def __init__(self, payload):
        self.payload = payload
        super().__init__('memory://archive')

    def request(self, byte_range):
        if byte_range.startswith('-'):
            data = self.payload[-int(byte_range[1:]):]
        else:
            start, end = map(int, byte_range.split('-'))
            data = self.payload[start:end+1]
        self.transferred = getattr(self, 'transferred', 0)+len(data)
        return data, len(self.payload)


class RemoteZipTests(unittest.TestCase):
    def test_deflate_member_and_zip64_local_header(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            with archive.open('nested/data.bin','w',force_zip64=True) as target:
                target.write(b'public fixture'*1000)
        reader = MemoryZip(buffer.getvalue())
        self.assertEqual(reader.read('nested/data.bin'), b'public fixture'*1000)

    def test_corrupt_payload_is_rejected(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('data.bin',b'123456')
        reader = MemoryZip(buffer.getvalue())
        reader.entries['data.bin']['crc'] ^= 1
        with self.assertRaisesRegex(ValueError,'CRC'): reader.read('data.bin')

    def test_oversized_member_is_rejected_before_payload_read(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer,'w') as archive: archive.writestr('data.bin',b'1')
        reader = MemoryZip(buffer.getvalue())
        reader.entries['data.bin']['size'] = 33*1024*1024
        with self.assertRaisesRegex(ValueError,'oversized'): reader.read('data.bin')


if __name__ == '__main__': unittest.main()
