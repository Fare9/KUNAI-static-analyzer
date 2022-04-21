// CkUnixCompress.h: interface for the CkUnixCompress class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkUnixCompress_H
#define _CkUnixCompress_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkTask;
class CkByteData;
class CkBaseProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkUnixCompress
class CK_VISIBLE_PUBLIC CkUnixCompress  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkUnixCompress(const CkUnixCompress &);
	CkUnixCompress &operator=(const CkUnixCompress &);

    public:
	CkUnixCompress(void);
	virtual ~CkUnixCompress(void);

	static CkUnixCompress *createNew(void);
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
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	bool get_AbortCurrent(void);
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	void put_AbortCurrent(bool newVal);

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
	// Compresses a file to create a UNIX compressed file (.Z). This compression uses
	// the LZW (Lempel-Ziv-Welch) compression algorithm.
	bool CompressFile(const char *inFilename, const char *destPath);

	// Compresses a file to create a UNIX compressed file (.Z). This compression uses
	// the LZW (Lempel-Ziv-Welch) compression algorithm.
	CkTask *CompressFileAsync(const char *inFilename, const char *destPath);


	// Unix compresses a file to an in-memory image of a .Z file. (This compression
	// uses the LZW (Lempel-Ziv-Welch) compression algorithm.)
	bool CompressFileToMem(const char *inFilename, CkByteData &outData);

	// Unix compresses a file to an in-memory image of a .Z file. (This compression
	// uses the LZW (Lempel-Ziv-Welch) compression algorithm.)
	CkTask *CompressFileToMemAsync(const char *inFilename);


	// Compresses in-memory data to an in-memory image of a .Z file. (This compression
	// uses the LZW (Lempel-Ziv-Welch) compression algorithm.)
	bool CompressMemory(CkByteData &inData, CkByteData &outData);


	// Unix compresses and creates a .Z file from in-memory data. (This compression
	// uses the LZW (Lempel-Ziv-Welch) compression algorithm.)
	bool CompressMemToFile(CkByteData &inData, const char *destPath);


	// Compresses a string to an in-memory image of a .Z file. Prior to compression,
	// the string is converted to the character encoding specified by charset. The charset
	// can be any one of a large number of character encodings, such as "utf-8",
	// "iso-8859-1", "Windows-1252", "ucs-2", etc.
	bool CompressString(const char *inStr, const char *charset, CkByteData &outBytes);


	// Unix compresses and creates a .Z file from string data. The charset specifies the
	// character encoding used for the byte representation of the characters when
	// compressed. The charset can be any one of a large number of character encodings,
	// such as "utf-8", "iso-8859-1", "Windows-1252", "ucs-2", etc.
	bool CompressStringToFile(const char *inStr, const char *charset, const char *destPath);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// Uncompresses a .Z file. (This compression uses the LZW (Lempel-Ziv-Welch)
	// compression algorithm.)
	bool UncompressFile(const char *inFilename, const char *destPath);

	// Uncompresses a .Z file. (This compression uses the LZW (Lempel-Ziv-Welch)
	// compression algorithm.)
	CkTask *UncompressFileAsync(const char *inFilename, const char *destPath);


	// Uncompresses a .Z file directly to memory. (This compression uses the LZW
	// (Lempel-Ziv-Welch) compression algorithm.)
	bool UncompressFileToMem(const char *inFilename, CkByteData &outData);

	// Uncompresses a .Z file directly to memory. (This compression uses the LZW
	// (Lempel-Ziv-Welch) compression algorithm.)
	CkTask *UncompressFileToMemAsync(const char *inFilename);


	// Uncompresses a .Z file that contains a text file. The contents of the text file
	// are returned as a string. The character encoding of the text file is specified
	// by charset. Typical charsets are "iso-8859-1", "utf-8", "windows-1252",
	// "shift_JIS", "big5", etc.
	bool UncompressFileToString(const char *zFilename, const char *charset, CkString &outStr);

	// Uncompresses a .Z file that contains a text file. The contents of the text file
	// are returned as a string. The character encoding of the text file is specified
	// by charset. Typical charsets are "iso-8859-1", "utf-8", "windows-1252",
	// "shift_JIS", "big5", etc.
	const char *uncompressFileToString(const char *zFilename, const char *charset);
	// Uncompresses a .Z file that contains a text file. The contents of the text file
	// are returned as a string. The character encoding of the text file is specified
	// by charset. Typical charsets are "iso-8859-1", "utf-8", "windows-1252",
	// "shift_JIS", "big5", etc.
	CkTask *UncompressFileToStringAsync(const char *zFilename, const char *charset);


	// Uncompresses from an in-memory image of a .Z file directly into memory. (This
	// compression uses the LZW (Lempel-Ziv-Welch) compression algorithm.)
	bool UncompressMemory(CkByteData &inData, CkByteData &outData);


	// Uncompresses from an in-memory image of a .Z file to a file. (This compression
	// uses the LZW (Lempel-Ziv-Welch) compression algorithm.)
	bool UncompressMemToFile(CkByteData &inData, const char *destPath);


	// Uncompresses from an in-memory image of a .Z file directly to a string. The charset
	// specifies the character encoding used to interpret the bytes resulting from the
	// decompression. The charset can be any one of a large number of character encodings,
	// such as "utf-8", "iso-8859-1", "Windows-1252", "ucs-2", etc.
	bool UncompressString(CkByteData &inCompressedData, const char *charset, CkString &outStr);

	// Uncompresses from an in-memory image of a .Z file directly to a string. The charset
	// specifies the character encoding used to interpret the bytes resulting from the
	// decompression. The charset can be any one of a large number of character encodings,
	// such as "utf-8", "iso-8859-1", "Windows-1252", "ucs-2", etc.
	const char *uncompressString(CkByteData &inCompressedData, const char *charset);

	// Unlocks the component allowing for the full functionality to be used.
	bool UnlockComponent(const char *unlockCode);


	// Unpacks a .tar.Z file. The decompress and untar occur in streaming mode. There
	// are no temporary files and the memory footprint is constant (and small) while
	// untarring.
	bool UnTarZ(const char *zFilename, const char *destDir, bool bNoAbsolute);

	// Unpacks a .tar.Z file. The decompress and untar occur in streaming mode. There
	// are no temporary files and the memory footprint is constant (and small) while
	// untarring.
	CkTask *UnTarZAsync(const char *zFilename, const char *destDir, bool bNoAbsolute);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
