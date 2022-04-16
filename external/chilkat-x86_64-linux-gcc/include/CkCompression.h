// CkCompression.h: interface for the CkCompression class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkCompression_H
#define _CkCompression_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkTask;
class CkByteData;
class CkBinData;
class CkStringBuilder;
class CkStream;
class CkBaseProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkCompression
class CK_VISIBLE_PUBLIC CkCompression  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkCompression(const CkCompression &);
	CkCompression &operator=(const CkCompression &);

    public:
	CkCompression(void);
	virtual ~CkCompression(void);

	static CkCompression *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	CkBaseProgress *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgress *progress);


	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The compression algorithm to be used. Should be set to either "ppmd", "deflate",
	// "zlib", "bzip2", or "lzw".
	// 
	// Note: The PPMD compression algorithm is only available for 32-bit builds. It is
	// not available for 64-bit implementations. Also, this PPMD implementation is the
	// "J" variant.
	// 
	void get_Algorithm(CkString &str);
	// The compression algorithm to be used. Should be set to either "ppmd", "deflate",
	// "zlib", "bzip2", or "lzw".
	// 
	// Note: The PPMD compression algorithm is only available for 32-bit builds. It is
	// not available for 64-bit implementations. Also, this PPMD implementation is the
	// "J" variant.
	// 
	const char *algorithm(void);
	// The compression algorithm to be used. Should be set to either "ppmd", "deflate",
	// "zlib", "bzip2", or "lzw".
	// 
	// Note: The PPMD compression algorithm is only available for 32-bit builds. It is
	// not available for 64-bit implementations. Also, this PPMD implementation is the
	// "J" variant.
	// 
	void put_Algorithm(const char *newVal);

	// Only applies to methods that compress or decompress strings. This specifies the
	// character encoding that the string should be converted to before compression.
	// Many programming languages use Unicode (2 bytes per char) for representing
	// characters. This property allows for the string to be converted to a 1-byte per
	// char encoding prior to compression.
	void get_Charset(CkString &str);
	// Only applies to methods that compress or decompress strings. This specifies the
	// character encoding that the string should be converted to before compression.
	// Many programming languages use Unicode (2 bytes per char) for representing
	// characters. This property allows for the string to be converted to a 1-byte per
	// char encoding prior to compression.
	const char *charset(void);
	// Only applies to methods that compress or decompress strings. This specifies the
	// character encoding that the string should be converted to before compression.
	// Many programming languages use Unicode (2 bytes per char) for representing
	// characters. This property allows for the string to be converted to a 1-byte per
	// char encoding prior to compression.
	void put_Charset(const char *newVal);

	// This property allows for customization of the compression level for the
	// "deflate" and "zlib" compression algoirthms. ("zlib" is just the deflate
	// algorithm with a zlib header.) A value of 0 = no compression, while 9 = maximum
	// compression. The default is 6.
	int get_DeflateLevel(void);
	// This property allows for customization of the compression level for the
	// "deflate" and "zlib" compression algoirthms. ("zlib" is just the deflate
	// algorithm with a zlib header.) A value of 0 = no compression, while 9 = maximum
	// compression. The default is 6.
	void put_DeflateLevel(int newVal);

	// Controls the encoding expected by methods ending in "ENC", such as
	// CompressBytesENC. Valid values are "base64", "hex", "url", and
	// "quoted-printable". Compression methods ending in ENC return the binary
	// compressed data as an encoded string using this encoding. Decompress methods
	// expect the input string to be this encoding.
	void get_EncodingMode(CkString &str);
	// Controls the encoding expected by methods ending in "ENC", such as
	// CompressBytesENC. Valid values are "base64", "hex", "url", and
	// "quoted-printable". Compression methods ending in ENC return the binary
	// compressed data as an encoded string using this encoding. Decompress methods
	// expect the input string to be this encoding.
	const char *encodingMode(void);
	// Controls the encoding expected by methods ending in "ENC", such as
	// CompressBytesENC. Valid values are "base64", "hex", "url", and
	// "quoted-printable". Compression methods ending in ENC return the binary
	// compressed data as an encoded string using this encoding. Decompress methods
	// expect the input string to be this encoding.
	void put_EncodingMode(const char *newVal);

	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any method call prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	int get_HeartbeatMs(void);
	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any method call prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	void put_HeartbeatMs(int newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Large amounts of binary byte data may be compressed in chunks by first calling
	// BeginCompressBytes, followed by 0 or more calls to MoreCompressedBytes, and
	// ending with a final call to EndCompressBytes. Each call returns 0 or more bytes
	// of compressed data which may be output to a compressed data stream (such as a
	// file, socket, etc.).
	bool BeginCompressBytes(CkByteData &data, CkByteData &outData);

	// Large amounts of binary byte data may be compressed in chunks by first calling
	// BeginCompressBytes, followed by 0 or more calls to MoreCompressedBytes, and
	// ending with a final call to EndCompressBytes. Each call returns 0 or more bytes
	// of compressed data which may be output to a compressed data stream (such as a
	// file, socket, etc.).
	CkTask *BeginCompressBytesAsync(CkByteData &data);


	// Large amounts of binary byte data may be compressed in chunks by first calling
	// BeginCompressBytes, followed by 0 or more calls to MoreCompressedBytes, and
	// ending with a final call to EndCompressBytes. Each call returns 0 or more bytes
	// of compressed data which may be output to a compressed data stream (such as a
	// file, socket, etc.).
	bool BeginCompressBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);


	// Large amounts of binary byte data may be compressed in chunks by first calling
	// BeginCompressBytesENC, followed by 0 or more calls to MoreCompressedBytesENC,
	// and ending with a final call to EndCompressBytesENC. Each call returns 0 or more
	// characters of compressed data (encoded as a string according to the EncodingMode
	// property setting) which may be output to a compressed data stream (such as a
	// file, socket, etc.).
	bool BeginCompressBytesENC(CkByteData &data, CkString &outStr);

	// Large amounts of binary byte data may be compressed in chunks by first calling
	// BeginCompressBytesENC, followed by 0 or more calls to MoreCompressedBytesENC,
	// and ending with a final call to EndCompressBytesENC. Each call returns 0 or more
	// characters of compressed data (encoded as a string according to the EncodingMode
	// property setting) which may be output to a compressed data stream (such as a
	// file, socket, etc.).
	const char *beginCompressBytesENC(CkByteData &data);
	// Large amounts of binary byte data may be compressed in chunks by first calling
	// BeginCompressBytesENC, followed by 0 or more calls to MoreCompressedBytesENC,
	// and ending with a final call to EndCompressBytesENC. Each call returns 0 or more
	// characters of compressed data (encoded as a string according to the EncodingMode
	// property setting) which may be output to a compressed data stream (such as a
	// file, socket, etc.).
	CkTask *BeginCompressBytesENCAsync(CkByteData &data);


	// Large amounts of string data may be compressed in chunks by first calling
	// BeginCompressString, followed by 0 or more calls to MoreCompressedString, and
	// ending with a final call to EndCompressString. Each call returns 0 or more bytes
	// of compressed data which may be output to a compressed data stream (such as a
	// file, socket, etc.).
	bool BeginCompressString(const char *str, CkByteData &outData);

	// Large amounts of string data may be compressed in chunks by first calling
	// BeginCompressString, followed by 0 or more calls to MoreCompressedString, and
	// ending with a final call to EndCompressString. Each call returns 0 or more bytes
	// of compressed data which may be output to a compressed data stream (such as a
	// file, socket, etc.).
	CkTask *BeginCompressStringAsync(const char *str);


	// Large amounts of string data may be compressed in chunks by first calling
	// BeginCompressStringENC, followed by 0 or more calls to MoreCompressedStringENC,
	// and ending with a final call to EndCompressStringENC. Each call returns 0 or
	// more characters of compressed data (encoded as a string according to the
	// EncodingMode property setting) which may be output to a compressed data stream
	// (such as a file, socket, etc.).
	bool BeginCompressStringENC(const char *str, CkString &outStr);

	// Large amounts of string data may be compressed in chunks by first calling
	// BeginCompressStringENC, followed by 0 or more calls to MoreCompressedStringENC,
	// and ending with a final call to EndCompressStringENC. Each call returns 0 or
	// more characters of compressed data (encoded as a string according to the
	// EncodingMode property setting) which may be output to a compressed data stream
	// (such as a file, socket, etc.).
	const char *beginCompressStringENC(const char *str);
	// Large amounts of string data may be compressed in chunks by first calling
	// BeginCompressStringENC, followed by 0 or more calls to MoreCompressedStringENC,
	// and ending with a final call to EndCompressStringENC. Each call returns 0 or
	// more characters of compressed data (encoded as a string according to the
	// EncodingMode property setting) which may be output to a compressed data stream
	// (such as a file, socket, etc.).
	CkTask *BeginCompressStringENCAsync(const char *str);


	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressBytes, followed by 0 or more calls to MoreDecompressedBytes, and
	// ending with a final call to EndDecompressBytes. Each call returns 0 or more
	// bytes of decompressed data.
	bool BeginDecompressBytes(CkByteData &data, CkByteData &outData);

	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressBytes, followed by 0 or more calls to MoreDecompressedBytes, and
	// ending with a final call to EndDecompressBytes. Each call returns 0 or more
	// bytes of decompressed data.
	CkTask *BeginDecompressBytesAsync(CkByteData &data);


	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressBytes, followed by 0 or more calls to MoreDecompressedBytes, and
	// ending with a final call to EndDecompressBytes. Each call returns 0 or more
	// bytes of decompressed data.
	bool BeginDecompressBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);


	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressBytesENC, followed by 0 or more calls to
	// MoreDecompressedBytesENC, and ending with a final call to EndDecompressBytesENC.
	// Each call returns 0 or more bytes of decompressed data.
	// 
	bool BeginDecompressBytesENC(const char *str, CkByteData &outData);

	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressBytesENC, followed by 0 or more calls to
	// MoreDecompressedBytesENC, and ending with a final call to EndDecompressBytesENC.
	// Each call returns 0 or more bytes of decompressed data.
	// 
	CkTask *BeginDecompressBytesENCAsync(const char *str);


	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressString, followed by 0 or more calls to MoreDecompressedString,
	// and ending with a final call to EndDecompressString. Each call returns 0 or more
	// characters of decompressed text.
	bool BeginDecompressString(CkByteData &data, CkString &outStr);

	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressString, followed by 0 or more calls to MoreDecompressedString,
	// and ending with a final call to EndDecompressString. Each call returns 0 or more
	// characters of decompressed text.
	const char *beginDecompressString(CkByteData &data);
	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressString, followed by 0 or more calls to MoreDecompressedString,
	// and ending with a final call to EndDecompressString. Each call returns 0 or more
	// characters of decompressed text.
	CkTask *BeginDecompressStringAsync(CkByteData &data);


	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressStringENC, followed by 0 or more calls to
	// MoreDecompressedStringENC, and ending with a final call to
	// EndDecompressStringENC. Each call returns 0 or more characters of decompressed
	// text.
	// 
	bool BeginDecompressStringENC(const char *str, CkString &outStr);

	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressStringENC, followed by 0 or more calls to
	// MoreDecompressedStringENC, and ending with a final call to
	// EndDecompressStringENC. Each call returns 0 or more characters of decompressed
	// text.
	// 
	const char *beginDecompressStringENC(const char *str);
	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// A compressed data stream may be decompressed in chunks by first calling
	// BeginDecompressStringENC, followed by 0 or more calls to
	// MoreDecompressedStringENC, and ending with a final call to
	// EndDecompressStringENC. Each call returns 0 or more characters of decompressed
	// text.
	// 
	CkTask *BeginDecompressStringENCAsync(const char *str);


	// Compresses the data contained in a BinData object.
	bool CompressBd(CkBinData &binData);

	// Compresses the data contained in a BinData object.
	CkTask *CompressBdAsync(CkBinData &binData);


	// Compresses byte data.
	bool CompressBytes(CkByteData &data, CkByteData &outData);

	// Compresses byte data.
	CkTask *CompressBytesAsync(CkByteData &data);


	// Compresses byte data.
	bool CompressBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);


	// Compresses bytes and returns the compressed data encoded to a string. The
	// encoding (hex, base64, etc.) is determined by the EncodingMode property setting.
	bool CompressBytesENC(CkByteData &data, CkString &outStr);

	// Compresses bytes and returns the compressed data encoded to a string. The
	// encoding (hex, base64, etc.) is determined by the EncodingMode property setting.
	const char *compressBytesENC(CkByteData &data);
	// Compresses bytes and returns the compressed data encoded to a string. The
	// encoding (hex, base64, etc.) is determined by the EncodingMode property setting.
	CkTask *CompressBytesENCAsync(CkByteData &data);


	// Performs file-to-file compression. Files of any size may be compressed because
	// the file is compressed internally in streaming mode.
	bool CompressFile(const char *srcPath, const char *destPath);

	// Performs file-to-file compression. Files of any size may be compressed because
	// the file is compressed internally in streaming mode.
	CkTask *CompressFileAsync(const char *srcPath, const char *destPath);


	// Compresses the contents of sb and appends the compressed bytes to binData.
	bool CompressSb(CkStringBuilder &sb, CkBinData &binData);

	// Compresses the contents of sb and appends the compressed bytes to binData.
	CkTask *CompressSbAsync(CkStringBuilder &sb, CkBinData &binData);


	// Compresses a stream. Internally, the strm's source is read, compressed, and the
	// compressed data written to the strm's sink. It does this in streaming fashion.
	// Extremely large or even infinite streams can be compressed with stable ungrowing
	// memory usage.
	bool CompressStream(CkStream &strm);

	// Compresses a stream. Internally, the strm's source is read, compressed, and the
	// compressed data written to the strm's sink. It does this in streaming fashion.
	// Extremely large or even infinite streams can be compressed with stable ungrowing
	// memory usage.
	CkTask *CompressStreamAsync(CkStream &strm);


	// Compresses a string.
	bool CompressString(const char *str, CkByteData &outData);

	// Compresses a string.
	CkTask *CompressStringAsync(const char *str);


	// Compresses a string and returns the compressed data encoded to a string. The
	// output encoding (hex, base64, etc.) is determined by the EncodingMode property
	// setting.
	bool CompressStringENC(const char *str, CkString &outStr);

	// Compresses a string and returns the compressed data encoded to a string. The
	// output encoding (hex, base64, etc.) is determined by the EncodingMode property
	// setting.
	const char *compressStringENC(const char *str);
	// Compresses a string and returns the compressed data encoded to a string. The
	// output encoding (hex, base64, etc.) is determined by the EncodingMode property
	// setting.
	CkTask *CompressStringENCAsync(const char *str);


	// Decompresses the data contained in a BinData object.
	bool DecompressBd(CkBinData &binData);

	// Decompresses the data contained in a BinData object.
	CkTask *DecompressBdAsync(CkBinData &binData);


	// The opposite of CompressBytes.
	bool DecompressBytes(CkByteData &data, CkByteData &outData);

	// The opposite of CompressBytes.
	CkTask *DecompressBytesAsync(CkByteData &data);


	// The opposite of CompressBytes2.
	bool DecompressBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);


	// The opposite of CompressBytesENC. encodedCompressedData contains the compressed data as an
	// encoded string (hex, base64, etc) as specified by the EncodingMode property
	// setting.
	bool DecompressBytesENC(const char *encodedCompressedData, CkByteData &outData);

	// The opposite of CompressBytesENC. encodedCompressedData contains the compressed data as an
	// encoded string (hex, base64, etc) as specified by the EncodingMode property
	// setting.
	CkTask *DecompressBytesENCAsync(const char *encodedCompressedData);


	// Performs file-to-file decompression (the opposite of CompressFile). Internally
	// the file is decompressed in streaming mode which allows files of any size to be
	// decompressed.
	bool DecompressFile(const char *srcPath, const char *destPath);

	// Performs file-to-file decompression (the opposite of CompressFile). Internally
	// the file is decompressed in streaming mode which allows files of any size to be
	// decompressed.
	CkTask *DecompressFileAsync(const char *srcPath, const char *destPath);


	// Decompresses the contents of binData and appends the decompressed string to sb.
	bool DecompressSb(CkBinData &binData, CkStringBuilder &sb);

	// Decompresses the contents of binData and appends the decompressed string to sb.
	CkTask *DecompressSbAsync(CkBinData &binData, CkStringBuilder &sb);


	// Decompresses a stream. Internally, the strm's source is read, decompressed, and
	// the decompressed data written to the strm's sink. It does this in streaming
	// fashion. Extremely large or even infinite streams can be decompressed with
	// stable ungrowing memory usage.
	bool DecompressStream(CkStream &strm);

	// Decompresses a stream. Internally, the strm's source is read, decompressed, and
	// the decompressed data written to the strm's sink. It does this in streaming
	// fashion. Extremely large or even infinite streams can be decompressed with
	// stable ungrowing memory usage.
	CkTask *DecompressStreamAsync(CkStream &strm);


	// Takes compressed bytes, decompresses, and returns the resulting string.
	bool DecompressString(CkByteData &data, CkString &outStr);

	// Takes compressed bytes, decompresses, and returns the resulting string.
	const char *decompressString(CkByteData &data);
	// Takes compressed bytes, decompresses, and returns the resulting string.
	CkTask *DecompressStringAsync(CkByteData &data);


	// The opposite of CompressStringENC. encodedCompressedData contains the compressed data as an
	// encoded string (hex, base64, etc) as specified by the EncodingMode property
	// setting.
	bool DecompressStringENC(const char *encodedCompressedData, CkString &outStr);

	// The opposite of CompressStringENC. encodedCompressedData contains the compressed data as an
	// encoded string (hex, base64, etc) as specified by the EncodingMode property
	// setting.
	const char *decompressStringENC(const char *encodedCompressedData);
	// The opposite of CompressStringENC. encodedCompressedData contains the compressed data as an
	// encoded string (hex, base64, etc) as specified by the EncodingMode property
	// setting.
	CkTask *DecompressStringENCAsync(const char *encodedCompressedData);


	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressBytes)
	// 
	bool EndCompressBytes(CkByteData &outData);

	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressBytes)
	// 
	CkTask *EndCompressBytesAsync(void);


	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressBytesENC)
	// 
	bool EndCompressBytesENC(CkString &outStr);

	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressBytesENC)
	// 
	const char *endCompressBytesENC(void);
	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressBytesENC)
	// 
	CkTask *EndCompressBytesENCAsync(void);


	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressString)
	// 
	bool EndCompressString(CkByteData &outData);

	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressString)
	// 
	CkTask *EndCompressStringAsync(void);


	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressStringENC)
	// 
	bool EndCompressStringENC(CkString &outStr);

	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressStringENC)
	// 
	const char *endCompressStringENC(void);
	// Must be callled to finalize a compression stream. Returns any remaining
	// (buffered) compressed data.
	// 
	// (See BeginCompressStringENC)
	// 
	CkTask *EndCompressStringENCAsync(void);


	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// (See BeginDecompressBytes)
	// 
	bool EndDecompressBytes(CkByteData &outData);

	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// (See BeginDecompressBytes)
	// 
	CkTask *EndDecompressBytesAsync(void);


	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressBytesENC)
	// 
	bool EndDecompressBytesENC(CkByteData &outData);

	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressBytesENC)
	// 
	CkTask *EndDecompressBytesENCAsync(void);


	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// (See BeginDecompressString)
	// 
	bool EndDecompressString(CkString &outStr);

	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// (See BeginDecompressString)
	// 
	const char *endDecompressString(void);
	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// (See BeginDecompressString)
	// 
	CkTask *EndDecompressStringAsync(void);


	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressStringENC)
	// 
	bool EndDecompressStringENC(CkString &outStr);

	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressStringENC)
	// 
	const char *endDecompressStringENC(void);
	// Called to finalize the decompression stream and return any remaining (buffered)
	// decompressed data.
	// 
	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressStringENC)
	// 
	CkTask *EndDecompressStringENCAsync(void);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// (See BeginCompressBytes)
	bool MoreCompressBytes(CkByteData &data, CkByteData &outData);

	// (See BeginCompressBytes)
	CkTask *MoreCompressBytesAsync(CkByteData &data);


	// (See BeginCompressBytes2)
	bool MoreCompressBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);


	// (See BeginCompressBytesENC)
	bool MoreCompressBytesENC(CkByteData &data, CkString &outStr);

	// (See BeginCompressBytesENC)
	const char *moreCompressBytesENC(CkByteData &data);
	// (See BeginCompressBytesENC)
	CkTask *MoreCompressBytesENCAsync(CkByteData &data);


	// (See BeginCompressString)
	bool MoreCompressString(const char *str, CkByteData &outData);

	// (See BeginCompressString)
	CkTask *MoreCompressStringAsync(const char *str);


	// (See BeginCompressStringENC)
	bool MoreCompressStringENC(const char *str, CkString &outStr);

	// (See BeginCompressStringENC)
	const char *moreCompressStringENC(const char *str);
	// (See BeginCompressStringENC)
	CkTask *MoreCompressStringENCAsync(const char *str);


	// (See BeginDecompressBytes)
	bool MoreDecompressBytes(CkByteData &data, CkByteData &outData);

	// (See BeginDecompressBytes)
	CkTask *MoreDecompressBytesAsync(CkByteData &data);


	// (See BeginDecompressBytes2)
	bool MoreDecompressBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);


	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressBytesENC)
	// 
	bool MoreDecompressBytesENC(const char *str, CkByteData &outData);

	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressBytesENC)
	// 
	CkTask *MoreDecompressBytesENCAsync(const char *str);


	// (See BeginDecompressString)
	bool MoreDecompressString(CkByteData &data, CkString &outStr);

	// (See BeginDecompressString)
	const char *moreDecompressString(CkByteData &data);
	// (See BeginDecompressString)
	CkTask *MoreDecompressStringAsync(CkByteData &data);


	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressStringENC)
	// 
	bool MoreDecompressStringENC(const char *str, CkString &outStr);

	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressStringENC)
	// 
	const char *moreDecompressStringENC(const char *str);
	// The input to this method is an encoded string containing compressed data. The
	// EncodingMode property should be set prior to calling this method. The input
	// string is decoded according to the EncodingMode (hex, base64, etc.) and then
	// decompressed.
	// 
	// (See BeginDecompressStringENC)
	// 
	CkTask *MoreDecompressStringENCAsync(const char *str);


	// Unlocks the component allowing for the full functionality to be used. The
	// component may be used fully-functional for the 1st 30-days after download by
	// passing an arbitrary string to this method. If for some reason you do not
	// receive the full 30-day trial, send email to support@chilkatsoft.com for a
	// temporary unlock code w/ an explicit expiration date. Upon purchase, a purchased
	// unlock code is provided which should replace the temporary/arbitrary string
	// passed to this method.
	bool UnlockComponent(const char *unlockCode);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
