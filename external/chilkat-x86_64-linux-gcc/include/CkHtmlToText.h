// CkHtmlToText.h: interface for the CkHtmlToText class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkHtmlToText_H
#define _CkHtmlToText_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkHtmlToText
class CK_VISIBLE_PUBLIC CkHtmlToText  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkHtmlToText(const CkHtmlToText &);
	CkHtmlToText &operator=(const CkHtmlToText &);

    public:
	CkHtmlToText(void);
	virtual ~CkHtmlToText(void);

	static CkHtmlToText *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// If true, then HTML entities are automatically decoded. For example _AMP_amp;
	// is automatically decoded to _AMP_. If this property is set to false, then HTML
	// entities are not decoded. The default value is true.
	bool get_DecodeHtmlEntities(void);
	// If true, then HTML entities are automatically decoded. For example _AMP_amp;
	// is automatically decoded to _AMP_. If this property is set to false, then HTML
	// entities are not decoded. The default value is true.
	void put_DecodeHtmlEntities(bool newVal);

	// Used to control wrapping of text. The default value is 80. When the text gets
	// close to this margin, the converter will try to break the line at a SPACE
	// character. Set this property to 0 for no right margin.
	int get_RightMargin(void);
	// Used to control wrapping of text. The default value is 80. When the text gets
	// close to this margin, the converter will try to break the line at a SPACE
	// character. Set this property to 0 for no right margin.
	void put_RightMargin(int newVal);

	// If false, then link URL's are preserved inline. For example, the following
	// HTML fragment:
	// 
	// _LT_p>Test _LT_a href="http://www.chilkatsoft.com/">chilkat_LT_/a>_LT_/p>
	// 
	// converts to:
	// Test chilkat _LT_http://www.chilkatsoft.com/>
	// If this property is true, the above HTML would convert to:
	// Test chilkat
	// The default value of this property is true.
	// 
	bool get_SuppressLinks(void);
	// If false, then link URL's are preserved inline. For example, the following
	// HTML fragment:
	// 
	// _LT_p>Test _LT_a href="http://www.chilkatsoft.com/">chilkat_LT_/a>_LT_/p>
	// 
	// converts to:
	// Test chilkat _LT_http://www.chilkatsoft.com/>
	// If this property is true, the above HTML would convert to:
	// Test chilkat
	// The default value of this property is true.
	// 
	void put_SuppressLinks(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Convenience method for reading a text file into a string. The character encoding
	// of the text file is specified by srcCharset. Valid values, such as "iso-8895-1" or
	// "utf-8" are listed at:List of Charsets
	// <http://blog.chilkatsoft.com/?p=463>.
	bool ReadFileToString(const char *filename, const char *srcCharset, CkString &outStr);

	// Convenience method for reading a text file into a string. The character encoding
	// of the text file is specified by srcCharset. Valid values, such as "iso-8895-1" or
	// "utf-8" are listed at:List of Charsets
	// <http://blog.chilkatsoft.com/?p=463>.
	const char *readFileToString(const char *filename, const char *srcCharset);

	// Converts HTML to plain-text.
	bool ToText(const char *html, CkString &outStr);

	// Converts HTML to plain-text.
	const char *toText(const char *html);

	// Unlocks the component. An arbitrary unlock code may be passed to automatically
	// begin a 30-day trial.
	// 
	// This class is included with the Chilkat HTML-to-XML conversion component
	// license.
	// 
	bool UnlockComponent(const char *code);


	// Convenience method for saving a string to a file. The character encoding of the
	// output text file is specified by charset (the string is converted to this charset
	// when writing). Valid values, such as "iso-8895-1" or "utf-8" are listed at:List
	// of Charsets
	// <http://blog.chilkatsoft.com/?p=463>.
	bool WriteStringToFile(const char *stringToWrite, const char *filename, const char *charset);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
