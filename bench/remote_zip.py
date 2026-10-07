"""Bounded HTTP-range ZIP/ZIP64 reader. Never falls back to downloading an archive."""
import binascii
import struct
import urllib.request
import zlib


class RemoteZip:
    def __init__(self, url):
        self.url = url
        self.transferred = 0
        tail, total = self.request('-65536')
        self.total = total
        end = tail.rfind(b'PK\x05\x06')
        if end < 0:
            raise ValueError('Missing ZIP end directory.')
        eocd = struct.unpack('<4s4H2LH', tail[end:end+22])
        size, offset = eocd[5:7]
        if offset == 0xffffffff or size == 0xffffffff or eocd[4] == 65535:
            locator = tail.rfind(b'PK\x06\x07', 0, end)
            if locator < 0:
                raise ValueError('Missing ZIP64 locator.')
            zip64_offset = struct.unpack('<4sLQL', tail[locator:locator+20])[2]
            record, _ = self.request(f'{zip64_offset}-{zip64_offset+55}')
            values = struct.unpack('<4sQ2H2L4Q', record)
            size, offset = values[8:10]
        if size > 32*1024*1024:
            raise ValueError('ZIP directory exceeds metadata budget.')
        directory, _ = self.request(f'{offset}-{offset+size-1}')
        self.entries = {}
        cursor = 0
        while cursor < len(directory):
            fields = struct.unpack_from('<4s6H3L5H2L', directory, cursor)
            if fields[0] != b'PK\x01\x02':
                raise ValueError('Invalid central directory header.')
            name_size, extra_size, comment_size = fields[10:13]
            start = cursor+46
            name = directory[start:start+name_size].decode('utf-8' if fields[3]&2048 else 'cp437')
            extra = directory[start+name_size:start+name_size+extra_size]
            compressed, expanded, local_offset = fields[8], fields[9], fields[16]
            index = 0
            while index+4 <= len(extra):
                kind, length = struct.unpack_from('<HH', extra, index)
                block = extra[index+4:index+4+length]
                if kind == 1:
                    at = 0
                    if expanded == 0xffffffff:
                        expanded = struct.unpack_from('<Q', block, at)[0]; at += 8
                    if compressed == 0xffffffff:
                        compressed = struct.unpack_from('<Q', block, at)[0]; at += 8
                    if local_offset == 0xffffffff:
                        local_offset = struct.unpack_from('<Q', block, at)[0]
                index += 4+length
            if name in self.entries:
                raise ValueError('Duplicate archive name.')
            self.entries[name] = dict(method=fields[4], flags=fields[3], crc=fields[7],
                                      compressed=compressed, size=expanded, offset=local_offset)
            cursor = start+name_size+extra_size+comment_size

    def request(self, byte_range):
        # Distinct cache keys prevent redirect/CDN caches reusing another range response.
        url = self.url+('&' if '?' in self.url else '?')+'range='+byte_range
        req = urllib.request.Request(url, headers={'Range': 'bytes='+byte_range})
        with urllib.request.urlopen(req, timeout=60) as response:
            header = response.headers.get('Content-Range', '')
            if response.status != 206 or not header.startswith('bytes '):
                raise ValueError('Server did not honour HTTP range; full download refused.')
            limits, total = header[6:].split('/')
            first, last = map(int, limits.split('-'))
            expected = last-first+1
            if not byte_range.startswith('-') and limits != byte_range:
                raise ValueError('Server returned a different byte range.')
            if expected > 32*1024*1024:
                raise ValueError('HTTP range exceeds memory budget.')
            data = response.read(expected+1)
            if len(data) != expected:
                raise ValueError('Incomplete byte range.')
            self.transferred += len(data)
            return data, int(total)

    def read(self, name):
        entry = self.entries[name]
        if entry['flags']&1 or max(entry['size'], entry['compressed']) > 32*1024*1024:
            raise ValueError('Encrypted/oversized member refused.')
        offset = entry['offset']
        header, _ = self.request(f'{offset}-{offset+29}')
        fields = struct.unpack('<4s5H3L2H', header)
        if fields[0] != b'PK\x03\x04':
            raise ValueError('Invalid local ZIP header.')
        start = offset+30+fields[-2]+fields[-1]
        raw, _ = self.request(f'{start}-{start+entry["compressed"]-1}')
        if entry['method'] == 0:
            result = raw
        elif entry['method'] == 8:
            decompressor = zlib.decompressobj(-15)
            result = decompressor.decompress(raw, entry['size']+1)
            if not decompressor.eof or decompressor.unconsumed_tail:
                raise ValueError('Invalid or oversized DEFLATE stream.')
        else:
            raise ValueError('Unsupported ZIP compression.')
        if len(result) != entry['size'] or binascii.crc32(result)&0xffffffff != entry['crc']:
            raise ValueError('Member size/CRC mismatch.')
        return result
