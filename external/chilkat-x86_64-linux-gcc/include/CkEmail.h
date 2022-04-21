// CkEmail.h: interface for the CkEmail class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkEmail_H
#define _CkEmail_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkBinData;
class CkByteData;
class CkCert;
class CkStringArray;
class CkDateTime;
class CkStringBuilder;
class CkCertChain;
class CkTask;
class CkPrivateKey;
class CkXmlCertVault;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkEmail
class CK_VISIBLE_PUBLIC CkEmail  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkEmail(const CkEmail &);
	CkEmail &operator=(const CkEmail &);

    public:
	CkEmail(void);
	virtual ~CkEmail(void);

	static CkEmail *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The body of the email. If the email has both HTML and plain-text bodies, this
	// property returns the HTML body. The GetHtmlBody and GetPlainTextBody methods can
	// be used to access a specific body. The HasHtmlBody and HasPlainTextBody methods
	// can be used to determine the presence of a body.
	void get_Body(CkString &str);
	// The body of the email. If the email has both HTML and plain-text bodies, this
	// property returns the HTML body. The GetHtmlBody and GetPlainTextBody methods can
	// be used to access a specific body. The HasHtmlBody and HasPlainTextBody methods
	// can be used to determine the presence of a body.
	const char *body(void);
	// The body of the email. If the email has both HTML and plain-text bodies, this
	// property returns the HTML body. The GetHtmlBody and GetPlainTextBody methods can
	// be used to access a specific body. The HasHtmlBody and HasPlainTextBody methods
	// can be used to determine the presence of a body.
	void put_Body(const char *newVal);

	// The "return-path" address of the email to be used when the email is sent.
	// Bounces (i.e. delivery status notifications, or DSN's) will go to this address.
	// 
	// Note: This is not the content of the "return-path" header for emails that are
	// downloaded from a POP3 or IMAP server. The BounceAddress is the email address to
	// be used in the process of sending the email via SMTP. (See the "SMTP Protocol in
	// a Nutshell" link below.) The BounceAddress is the email address passed in the
	// "MAIL FROM" SMTP command which becomes the "return-path" header in the email
	// when received.
	// 
	// Note: The Sender and BounceAddress properties are identical and perform the same
	// function. Setting the Sender property also sets the BounceAddress property, and
	// vice-versa. The reason for the duplication is that BounceAddress existed first,
	// and developers typically searched for a "Sender" property without realizing that
	// the BounceAddress property served this function.
	// 
	void get_BounceAddress(CkString &str);
	// The "return-path" address of the email to be used when the email is sent.
	// Bounces (i.e. delivery status notifications, or DSN's) will go to this address.
	// 
	// Note: This is not the content of the "return-path" header for emails that are
	// downloaded from a POP3 or IMAP server. The BounceAddress is the email address to
	// be used in the process of sending the email via SMTP. (See the "SMTP Protocol in
	// a Nutshell" link below.) The BounceAddress is the email address passed in the
	// "MAIL FROM" SMTP command which becomes the "return-path" header in the email
	// when received.
	// 
	// Note: The Sender and BounceAddress properties are identical and perform the same
	// function. Setting the Sender property also sets the BounceAddress property, and
	// vice-versa. The reason for the duplication is that BounceAddress existed first,
	// and developers typically searched for a "Sender" property without realizing that
	// the BounceAddress property served this function.
	// 
	const char *bounceAddress(void);
	// The "return-path" address of the email to be used when the email is sent.
	// Bounces (i.e. delivery status notifications, or DSN's) will go to this address.
	// 
	// Note: This is not the content of the "return-path" header for emails that are
	// downloaded from a POP3 or IMAP server. The BounceAddress is the email address to
	// be used in the process of sending the email via SMTP. (See the "SMTP Protocol in
	// a Nutshell" link below.) The BounceAddress is the email address passed in the
	// "MAIL FROM" SMTP command which becomes the "return-path" header in the email
	// when received.
	// 
	// Note: The Sender and BounceAddress properties are identical and perform the same
	// function. Setting the Sender property also sets the BounceAddress property, and
	// vice-versa. The reason for the duplication is that BounceAddress existed first,
	// and developers typically searched for a "Sender" property without realizing that
	// the BounceAddress property served this function.
	// 
	void put_BounceAddress(const char *newVal);

	// Sets the charset for the entire email. The header fields and plain-text/HTML
	// bodies will be converted and sent in this charset. (This includes parsing and
	// updating the HTML with the appropriate META tag specifying the charset.) All
	// formatting and encoding of the email MIME is handled automatically by the
	// Chilkat Mail component. If your application wants to send a Shift_JIS email, you
	// simply set the Charset property to "Shift_JIS". Note: If a charset property is
	// not explicitly set, the Chilkat component automatically detects the charset and
	// chooses the appropriate charset. If all characters are 7bit (i.e. us-ascii) the
	// charset is left blank. If the email contain a mix of languages such that no one
	// charset can be chosen, or if the language cannot be determined without
	// ambiguity, then the "utf-8" charset will be chosen.
	void get_Charset(CkString &str);
	// Sets the charset for the entire email. The header fields and plain-text/HTML
	// bodies will be converted and sent in this charset. (This includes parsing and
	// updating the HTML with the appropriate META tag specifying the charset.) All
	// formatting and encoding of the email MIME is handled automatically by the
	// Chilkat Mail component. If your application wants to send a Shift_JIS email, you
	// simply set the Charset property to "Shift_JIS". Note: If a charset property is
	// not explicitly set, the Chilkat component automatically detects the charset and
	// chooses the appropriate charset. If all characters are 7bit (i.e. us-ascii) the
	// charset is left blank. If the email contain a mix of languages such that no one
	// charset can be chosen, or if the language cannot be determined without
	// ambiguity, then the "utf-8" charset will be chosen.
	const char *charset(void);
	// Sets the charset for the entire email. The header fields and plain-text/HTML
	// bodies will be converted and sent in this charset. (This includes parsing and
	// updating the HTML with the appropriate META tag specifying the charset.) All
	// formatting and encoding of the email MIME is handled automatically by the
	// Chilkat Mail component. If your application wants to send a Shift_JIS email, you
	// simply set the Charset property to "Shift_JIS". Note: If a charset property is
	// not explicitly set, the Chilkat component automatically detects the charset and
	// chooses the appropriate charset. If all characters are 7bit (i.e. us-ascii) the
	// charset is left blank. If the email contain a mix of languages such that no one
	// charset can be chosen, or if the language cannot be determined without
	// ambiguity, then the "utf-8" charset will be chosen.
	void put_Charset(const char *newVal);

	// true if the email arrived encrypted and was successfully decrypted, otherwise
	// false. This property is only meaningful when the ReceivedEncrypted property is
	// equal to true.
	bool get_Decrypted(void);

	// The date/time from the "Date" header in UTC/GMT standard time. Use the LocalDate
	// property to get the local date and time. This property should no longer be used
	// because it will be removed in a future version. Use the GetDt method instead.
	void get_EmailDate(SYSTEMTIME &outSysTime);
	// The date/time from the "Date" header in UTC/GMT standard time. Use the LocalDate
	// property to get the local date and time. This property should no longer be used
	// because it will be removed in a future version. Use the GetDt method instead.
	void put_EmailDate(const SYSTEMTIME &sysTime);

	// The date/time from the "Date" header in the UTC/GMT timezone in RFC822 string
	// form.
	void get_EmailDateStr(CkString &str);
	// The date/time from the "Date" header in the UTC/GMT timezone in RFC822 string
	// form.
	const char *emailDateStr(void);
	// The date/time from the "Date" header in the UTC/GMT timezone in RFC822 string
	// form.
	void put_EmailDateStr(const char *newVal);

	// If the email was received encrypted, this contains the details of the
	// certificate used for encryption.
	void get_EncryptedBy(CkString &str);
	// If the email was received encrypted, this contains the details of the
	// certificate used for encryption.
	const char *encryptedBy(void);

	// Set this property to send an email to a list of recipients stored in a plain
	// text file. The file format is simple: one recipient per line, no comments
	// allowed, blank lines are ignored.Setting this property is equivalent to adding a
	// "CKX-FileDistList"header field to the email. Chilkat Mail treats header fields
	// beginning with "CKX-"specially in that these fields are never transmitted with
	// the email when sent. However, CKX fields are saved and restored when saving to
	// XML or loading from XML (or MIME). When sending an email containing a
	// "CKX-FileDistList"header field, Chilkat Mail will read the distribution list
	// file and send the email to each recipient. Emails can be sent individually, or
	// with BCC, 100 recipients at a time. (see the MailMan.SendIndividual property).
	void get_FileDistList(CkString &str);
	// Set this property to send an email to a list of recipients stored in a plain
	// text file. The file format is simple: one recipient per line, no comments
	// allowed, blank lines are ignored.Setting this property is equivalent to adding a
	// "CKX-FileDistList"header field to the email. Chilkat Mail treats header fields
	// beginning with "CKX-"specially in that these fields are never transmitted with
	// the email when sent. However, CKX fields are saved and restored when saving to
	// XML or loading from XML (or MIME). When sending an email containing a
	// "CKX-FileDistList"header field, Chilkat Mail will read the distribution list
	// file and send the email to each recipient. Emails can be sent individually, or
	// with BCC, 100 recipients at a time. (see the MailMan.SendIndividual property).
	const char *fileDistList(void);
	// Set this property to send an email to a list of recipients stored in a plain
	// text file. The file format is simple: one recipient per line, no comments
	// allowed, blank lines are ignored.Setting this property is equivalent to adding a
	// "CKX-FileDistList"header field to the email. Chilkat Mail treats header fields
	// beginning with "CKX-"specially in that these fields are never transmitted with
	// the email when sent. However, CKX fields are saved and restored when saving to
	// XML or loading from XML (or MIME). When sending an email containing a
	// "CKX-FileDistList"header field, Chilkat Mail will read the distribution list
	// file and send the email to each recipient. Emails can be sent individually, or
	// with BCC, 100 recipients at a time. (see the MailMan.SendIndividual property).
	void put_FileDistList(const char *newVal);

	// The combined name and email address of the sender, such as "John Smith" . This
	// is the content that will be placed in the From: header field. If the actual
	// sender is to be different, then set the Sender property to a different email
	// address.
	void get_From(CkString &str);
	// The combined name and email address of the sender, such as "John Smith" . This
	// is the content that will be placed in the From: header field. If the actual
	// sender is to be different, then set the Sender property to a different email
	// address.
	const char *ck_from(void);
	// The combined name and email address of the sender, such as "John Smith" . This
	// is the content that will be placed in the From: header field. If the actual
	// sender is to be different, then set the Sender property to a different email
	// address.
	void put_From(const char *newVal);

	// The email address of the sender.
	void get_FromAddress(CkString &str);
	// The email address of the sender.
	const char *fromAddress(void);
	// The email address of the sender.
	void put_FromAddress(const char *newVal);

	// The name of the sender.
	void get_FromName(CkString &str);
	// The name of the sender.
	const char *fromName(void);
	// The name of the sender.
	void put_FromName(const char *newVal);

	// The complete MIME header of the email.
	void get_Header(CkString &str);
	// The complete MIME header of the email.
	const char *header(void);

	// A read-only property that identifies the primary language group for the email.
	// Possible values are:
	// 
	//         "latin1" (for English and all Western European languages)
	//         "central" (for Central European languages such as Polish, Czech,
	//         Hungarian, etc.)
	//         "russian" (for Cyrillic languages)
	//         "greek"
	//         "turkish"
	//         "hebrew"
	//         "arabic"
	//         "thai"
	//         "vietnamese"
	//         "chinese"
	//         "japanese"
	//         "korean"
	//         "devanagari"
	//         "bengali"
	//         "gurmukhi"
	//         "gujarati"
	//         "oriya"
	//         "tamil"
	//         "telugu"
	//         "kannada"
	//         "malayalam"
	//         "sinhala"
	//         "lao"
	//         "tibetan"
	//         "myanmar"
	//         "georgian"
	//         "unknown"
	// 
	// The language group determination is made soley on the subject and
	// plain-text/HTML email bodies. Characters in the FROM, TO, CC, and other header
	// fields are not considered.
	// 
	// The primary determining factor is the characters found in the Subject header
	// field. For example, if an email contains Japanese in the Subject, but the body
	// contains Russian characters, it will be considered "japanese".
	// 
	// The language is determined by where the Unicode chars fall in various blocks in
	// the Unicode Basic Multilingual Plane. For more information, see the book
	// "Unicode Demystified" by Richard Gillam.
	// 
	void get_Language(CkString &str);
	// A read-only property that identifies the primary language group for the email.
	// Possible values are:
	// 
	//         "latin1" (for English and all Western European languages)
	//         "central" (for Central European languages such as Polish, Czech,
	//         Hungarian, etc.)
	//         "russian" (for Cyrillic languages)
	//         "greek"
	//         "turkish"
	//         "hebrew"
	//         "arabic"
	//         "thai"
	//         "vietnamese"
	//         "chinese"
	//         "japanese"
	//         "korean"
	//         "devanagari"
	//         "bengali"
	//         "gurmukhi"
	//         "gujarati"
	//         "oriya"
	//         "tamil"
	//         "telugu"
	//         "kannada"
	//         "malayalam"
	//         "sinhala"
	//         "lao"
	//         "tibetan"
	//         "myanmar"
	//         "georgian"
	//         "unknown"
	// 
	// The language group determination is made soley on the subject and
	// plain-text/HTML email bodies. Characters in the FROM, TO, CC, and other header
	// fields are not considered.
	// 
	// The primary determining factor is the characters found in the Subject header
	// field. For example, if an email contains Japanese in the Subject, but the body
	// contains Russian characters, it will be considered "japanese".
	// 
	// The language is determined by where the Unicode chars fall in various blocks in
	// the Unicode Basic Multilingual Plane. For more information, see the book
	// "Unicode Demystified" by Richard Gillam.
	// 
	const char *language(void);

	// The date/time found in the "Date" header field returned in the local timezone.
	// This property should no longer be used because it will be removed in a future
	// version. Use the GetDt method instead.
	void get_LocalDate(SYSTEMTIME &outSysTime);
	// The date/time found in the "Date" header field returned in the local timezone.
	// This property should no longer be used because it will be removed in a future
	// version. Use the GetDt method instead.
	void put_LocalDate(const SYSTEMTIME &sysTime);

	// The date/time found in the "Date" header field returned in the local timezone in
	// RFC822 string form.
	void get_LocalDateStr(CkString &str);
	// The date/time found in the "Date" header field returned in the local timezone in
	// RFC822 string form.
	const char *localDateStr(void);
	// The date/time found in the "Date" header field returned in the local timezone in
	// RFC822 string form.
	void put_LocalDateStr(const char *newVal);

	// Identifies the email software that sent the email.
	void get_Mailer(CkString &str);
	// Identifies the email software that sent the email.
	const char *mailer(void);
	// Identifies the email software that sent the email.
	void put_Mailer(const char *newVal);

	// The number of alternative bodies present in the email. An email that is not
	// "multipart/alternative"will return 0 alternatives. An email that is
	// "multipart/alternative" will return a number greater than or equal to 1.
	int get_NumAlternatives(void);

	// Returns the number of embedded emails. Some mail clients will embed an email
	// that is to be forwarded into a new email as a "message/rfc822" subpart of the
	// MIME message structure. This property tells how many emails have been embedded.
	// The original email can be retrieved by calling GetAttachedMessage.
	int get_NumAttachedMessages(void);

	// The number of attachments contained in the email.
	// 
	// Note: If an email is downloaded from an IMAP server without attachments, then
	// the number of attachments should be obtained by calling the IMAP object's
	// GetMailNumAttach method. This property indicates the actual number of
	// attachments already present within the email object.
	// 
	int get_NumAttachments(void);

	// The number of blind carbon-copy email recipients.
	int get_NumBcc(void);

	// The number of carbon-copy email recipients.
	int get_NumCC(void);

	// Returns the number of days old from the current system date/time. The email's
	// date is obtained from the "Date" header field. If the Date header field is
	// missing, or invalid, then -9999 is returned. A negative number may be returned
	// if the Date header field contains a future date/time. (However, -9999 represents
	// an error condition.)
	int get_NumDaysOld(void);

	// Returns the number of message/rfc822 parts contained within the multipart/digest
	// enclosure. If no multipart/digest enclosure exists, then this property has the
	// value of 0. The GetDigest method is called to get the Nth digest as an email
	// object.
	int get_NumDigests(void);

	// The number of header fields. When accessing a header field by index, the 1st
	// header field is at index 0, and the last is at NumHeaderFields-1. (Chilkat
	// indexing is always 0-based.)
	int get_NumHeaderFields(void);

	// The number of related items present in this email. Related items are typically
	// image files (JPEGs or GIFs) or style sheets (CSS files) that are included with
	// HTML formatted messages with internal "CID"hyperlinks.
	int get_NumRelatedItems(void);

	// Returns the number of replacement patterns previously set by calling the
	// SetReplacePattern method 1 or more times. If replacement patterns are set, the
	// email bodies and header fields are modified by applying the search/replacement
	// strings during the message sending process.
	int get_NumReplacePatterns(void);

	// (For multipart/report emails that have sub-parts with Content-Types such as
	// message/feedback-report.) Any MIME sub-part within the email that has a
	// Content-Type of "message/*", but is not a "message/rfc822", is considered to be
	// a "report" and is included in this count. (A "message/rfc822" is considered an
	// attached message and is handled by the NumAttachedMessages property and the
	// GetAttachedMessage method.) Any MIME sub-part having a Content-Type equal to
	// "text/rfc822-headers" is also considered to be a "report". The GetReport method
	// may be called to get the body content of each "report" contained within a
	// multipart/report email.
	int get_NumReports(void);

	// The number of direct email recipients.
	int get_NumTo(void);

	// Selects the hash algorithm for use within OAEP padding when encrypting email
	// using RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384", "sha512",
	void get_OaepHash(CkString &str);
	// Selects the hash algorithm for use within OAEP padding when encrypting email
	// using RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384", "sha512",
	const char *oaepHash(void);
	// Selects the hash algorithm for use within OAEP padding when encrypting email
	// using RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384", "sha512",
	void put_OaepHash(const char *newVal);

	// Selects the MGF hash algorithm for use within OAEP padding when encrypting email
	// using RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384", "sha512",
	void get_OaepMgfHash(CkString &str);
	// Selects the MGF hash algorithm for use within OAEP padding when encrypting email
	// using RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384", "sha512",
	const char *oaepMgfHash(void);
	// Selects the MGF hash algorithm for use within OAEP padding when encrypting email
	// using RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384", "sha512",
	void put_OaepMgfHash(const char *newVal);

	// Selects the RSA encryption scheme when encrypting email. The default value is
	// false, which selects RSAES_PKCS1-V1_5. If set to true, then RSAES_OAEP is
	// used.
	bool get_OaepPadding(void);
	// Selects the RSA encryption scheme when encrypting email. The default value is
	// false, which selects RSAES_PKCS1-V1_5. If set to true, then RSAES_OAEP is
	// used.
	void put_OaepPadding(bool newVal);

	// When true (the default) the methods to save email attachments and related
	// items will overwrite files if they already exist. If false, then the methods
	// that save email attachments and related items will append a string of 4
	// characters to create a unique filename if a file already exists. The filename of
	// the attachment (or related item) within the email object is updated and can be
	// retrieved by the program to determine the actual file(s) created.
	bool get_OverwriteExisting(void);
	// When true (the default) the methods to save email attachments and related
	// items will overwrite files if they already exist. If false, then the methods
	// that save email attachments and related items will append a string of 4
	// characters to create a unique filename if a file already exists. The filename of
	// the attachment (or related item) within the email object is updated and can be
	// retrieved by the program to determine the actual file(s) created.
	void put_OverwriteExisting(bool newVal);

	// When an email is sent encrypted (using PKCS7 public-key encryption), this
	// selects the underlying symmetric encryption algorithm. Possible values are:
	// "aes", "des", "3des", and "rc2". The default value is "aes".
	void get_Pkcs7CryptAlg(CkString &str);
	// When an email is sent encrypted (using PKCS7 public-key encryption), this
	// selects the underlying symmetric encryption algorithm. Possible values are:
	// "aes", "des", "3des", and "rc2". The default value is "aes".
	const char *pkcs7CryptAlg(void);
	// When an email is sent encrypted (using PKCS7 public-key encryption), this
	// selects the underlying symmetric encryption algorithm. Possible values are:
	// "aes", "des", "3des", and "rc2". The default value is "aes".
	void put_Pkcs7CryptAlg(const char *newVal);

	// When the email is sent encrypted (using PKCS7 public-key encryption), this
	// selects the key length of the underlying symmetric encryption algorithm. The
	// possible values allowed depend on the Pkcs7CryptAlg property. For "aes", the key
	// length may be 128, 192, or 256. For "3des" the key length must be 192. For "des"
	// the key length must be 40. For "rc2" the key length can be 40, 56, 64, or 128.
	int get_Pkcs7KeyLength(void);
	// When the email is sent encrypted (using PKCS7 public-key encryption), this
	// selects the key length of the underlying symmetric encryption algorithm. The
	// possible values allowed depend on the Pkcs7CryptAlg property. For "aes", the key
	// length may be 128, 192, or 256. For "3des" the key length must be 192. For "des"
	// the key length must be 40. For "rc2" the key length can be 40, 56, 64, or 128.
	void put_Pkcs7KeyLength(int newVal);

	// Only applies when building an email with non-English characters where the
	// charset is not explicitly set. The Chilkat email component will automatically
	// choose a charset based on the languages found within an email (if the charset is
	// not already specified within the MIME or explicitly specified by setting the
	// Charset property). The default charset chosen for each language is:
	// 
	// Chinese: gb2312
	// Japanese: shift_JIS
	// Korean: ks_c_5601-1987
	// Thai: windows-874
	// All others: iso-8859-*
	// 
	// This allows for charsets such as iso-2022-jp to be chosen instead of the
	// default. If the preferred charset does not apply to the situation, it is not
	// used. For example, if the preferred charset is iso-2022-jp, but the email
	// contains Greek characters, then the preferred charset is ignored.
	// 
	void get_PreferredCharset(CkString &str);
	// Only applies when building an email with non-English characters where the
	// charset is not explicitly set. The Chilkat email component will automatically
	// choose a charset based on the languages found within an email (if the charset is
	// not already specified within the MIME or explicitly specified by setting the
	// Charset property). The default charset chosen for each language is:
	// 
	// Chinese: gb2312
	// Japanese: shift_JIS
	// Korean: ks_c_5601-1987
	// Thai: windows-874
	// All others: iso-8859-*
	// 
	// This allows for charsets such as iso-2022-jp to be chosen instead of the
	// default. If the preferred charset does not apply to the situation, it is not
	// used. For example, if the preferred charset is iso-2022-jp, but the email
	// contains Greek characters, then the preferred charset is ignored.
	// 
	const char *preferredCharset(void);
	// Only applies when building an email with non-English characters where the
	// charset is not explicitly set. The Chilkat email component will automatically
	// choose a charset based on the languages found within an email (if the charset is
	// not already specified within the MIME or explicitly specified by setting the
	// Charset property). The default charset chosen for each language is:
	// 
	// Chinese: gb2312
	// Japanese: shift_JIS
	// Korean: ks_c_5601-1987
	// Thai: windows-874
	// All others: iso-8859-*
	// 
	// This allows for charsets such as iso-2022-jp to be chosen instead of the
	// default. If the preferred charset does not apply to the situation, it is not
	// used. For example, if the preferred charset is iso-2022-jp, but the email
	// contains Greek characters, then the preferred charset is ignored.
	// 
	void put_PreferredCharset(const char *newVal);

	// If true, then header fields added via the AddHeaderField or AddHeaderField2
	// methods are prepended to the top of the header as opposed to appended to the
	// bottom. The default value is false.
	bool get_PrependHeaders(void);
	// If true, then header fields added via the AddHeaderField or AddHeaderField2
	// methods are prepended to the top of the header as opposed to appended to the
	// bottom. The default value is false.
	void put_PrependHeaders(bool newVal);

	// true if this email was originally received with encryption, otherwise false.
	bool get_ReceivedEncrypted(void);

	// true if this email was originally received with a digital signature, otherwise
	// false.
	bool get_ReceivedSigned(void);

	// Sets the "Reply-To" header field to the specified email address.
	void get_ReplyTo(CkString &str);
	// Sets the "Reply-To" header field to the specified email address.
	const char *replyTo(void);
	// Sets the "Reply-To" header field to the specified email address.
	void put_ReplyTo(const char *newVal);

	// Set to true if you want the email to request a return-receipt when received by
	// the recipient. The default value is false.
	bool get_ReturnReceipt(void);
	// Set to true if you want the email to request a return-receipt when received by
	// the recipient. The default value is false.
	void put_ReturnReceipt(bool newVal);

	// Set to true if this email should be sent encrypted.
	bool get_SendEncrypted(void);
	// Set to true if this email should be sent encrypted.
	void put_SendEncrypted(bool newVal);

	// The sender's address for this email message.
	// 
	// This is the address of the actual sender acting on behalf of the author listed
	// in the From: field.
	// 
	// Note: The Sender and BounceAddress properties are identical and perform the same
	// function. Setting the Sender property also sets the BounceAddress property, and
	// vice-versa. The reason for the duplication is that BounceAddress existed first,
	// and developers typically searched for a "Sender" property without realizing that
	// the BounceAddress property served this function.
	// 
	// Important Note: This property does not contain the value of the "Sender" header
	// field, if one exists, for a received email. It is a property that is used when
	// sending email. To get the value of the "Sender" header field (which may not
	// always exist), call the GetHeaderField method instead.
	// 
	void get_Sender(CkString &str);
	// The sender's address for this email message.
	// 
	// This is the address of the actual sender acting on behalf of the author listed
	// in the From: field.
	// 
	// Note: The Sender and BounceAddress properties are identical and perform the same
	// function. Setting the Sender property also sets the BounceAddress property, and
	// vice-versa. The reason for the duplication is that BounceAddress existed first,
	// and developers typically searched for a "Sender" property without realizing that
	// the BounceAddress property served this function.
	// 
	// Important Note: This property does not contain the value of the "Sender" header
	// field, if one exists, for a received email. It is a property that is used when
	// sending email. To get the value of the "Sender" header field (which may not
	// always exist), call the GetHeaderField method instead.
	// 
	const char *sender(void);
	// The sender's address for this email message.
	// 
	// This is the address of the actual sender acting on behalf of the author listed
	// in the From: field.
	// 
	// Note: The Sender and BounceAddress properties are identical and perform the same
	// function. Setting the Sender property also sets the BounceAddress property, and
	// vice-versa. The reason for the duplication is that BounceAddress existed first,
	// and developers typically searched for a "Sender" property without realizing that
	// the BounceAddress property served this function.
	// 
	// Important Note: This property does not contain the value of the "Sender" header
	// field, if one exists, for a received email. It is a property that is used when
	// sending email. To get the value of the "Sender" header field (which may not
	// always exist), call the GetHeaderField method instead.
	// 
	void put_Sender(const char *newVal);

	// Set to true if this email should be sent with a digital signature.
	// 
	// Note: Please be aware of the Apple Mail bug explained in the web page at the
	// link below.
	// 
	bool get_SendSigned(void);
	// Set to true if this email should be sent with a digital signature.
	// 
	// Note: Please be aware of the Apple Mail bug explained in the web page at the
	// link below.
	// 
	void put_SendSigned(bool newVal);

	// true if the email was received with one or more digital signatures, and if all
	// the signatures were validated indicating that the email was not altered.
	// Otherwise this property is set to false. (This property is only meaningful
	// when the ReceivedSigned property is equal to true.)
	bool get_SignaturesValid(void);

	// If the email was received digitally signed, this property contains the fields of
	// the cert's SubjectDN.
	// 
	// For example: US, 60187, Illinois, Wheaton, 1719 E Forest Ave, "Chilkat Software,
	// Inc.", "Chilkat Software, Inc."
	// 
	// It is like the DN (Distinguished Name), but without the "AttrName=" before each
	// attribute.
	// 
	void get_SignedBy(CkString &str);
	// If the email was received digitally signed, this property contains the fields of
	// the cert's SubjectDN.
	// 
	// For example: US, 60187, Illinois, Wheaton, 1719 E Forest Ave, "Chilkat Software,
	// Inc.", "Chilkat Software, Inc."
	// 
	// It is like the DN (Distinguished Name), but without the "AttrName=" before each
	// attribute.
	// 
	const char *signedBy(void);

	// Selects the signature algorithm to be used when sending signed (PKCS7) email.
	// The default value is "PKCS1-v1_5". This can be set to "RSASSA-PSS" (or simply
	// "pss") to use the RSASSA-PSS signature scheme.
	// 
	// Note: This property only applies when signing with an RSA private key. It does
	// not apply for ECC or DSA private keys.
	// 
	void get_SigningAlg(CkString &str);
	// Selects the signature algorithm to be used when sending signed (PKCS7) email.
	// The default value is "PKCS1-v1_5". This can be set to "RSASSA-PSS" (or simply
	// "pss") to use the RSASSA-PSS signature scheme.
	// 
	// Note: This property only applies when signing with an RSA private key. It does
	// not apply for ECC or DSA private keys.
	// 
	const char *signingAlg(void);
	// Selects the signature algorithm to be used when sending signed (PKCS7) email.
	// The default value is "PKCS1-v1_5". This can be set to "RSASSA-PSS" (or simply
	// "pss") to use the RSASSA-PSS signature scheme.
	// 
	// Note: This property only applies when signing with an RSA private key. It does
	// not apply for ECC or DSA private keys.
	// 
	void put_SigningAlg(const char *newVal);

	// Selects the underlying hash algorithm used when sending signed (PKCS7) email.
	// Possible values are "sha1", "sha256", "sha384", "sha512", "md5", and "md2".
	void get_SigningHashAlg(CkString &str);
	// Selects the underlying hash algorithm used when sending signed (PKCS7) email.
	// Possible values are "sha1", "sha256", "sha384", "sha512", "md5", and "md2".
	const char *signingHashAlg(void);
	// Selects the underlying hash algorithm used when sending signed (PKCS7) email.
	// Possible values are "sha1", "sha256", "sha384", "sha512", "md5", and "md2".
	void put_SigningHashAlg(const char *newVal);

	// The size in bytes of the email, including all parts and attachments.
	// 
	// Note: This property is only valid if the full email was downloaded. If only the
	// header was downloaded, then this property will contain the size of just the
	// header.
	// 
	int get_Size(void);

	// The email subject.
	void get_Subject(CkString &str);
	// The email subject.
	const char *subject(void);
	// The email subject.
	void put_Subject(const char *newVal);

	// This is the unique ID assigned by the POP3 server. Emails can be retrieved or
	// deleted from the POP3 server via the UIDL. The header field for this property is
	// "X-UIDL".
	// 
	// Important: Emails downloaded via the IMAP protocol do not have UIDL's. UIDL's
	// are specific to the POP3 protocol. IMAP servers use UID's (notice the spelling
	// difference -- "UIDL" vs. "UID"). An email downloaded via the Chilkat IMAP
	// component will contain a "ckx-imap-uid" header field that holds either an IMAP
	// UID or sequence number. If the email was downloaded (from IMAP) using a UID,then
	// ckx-imap-uid contains a UID. If the email was downloaded by sequence number,
	// then it contains a sequence number (not unique and can change with a new
	// session). The "ckx-imap-isUid" header, which can have the value "YES" or "NO",
	// tells what ckx-imap-uid contains.
	// 
	void get_Uidl(CkString &str);
	// This is the unique ID assigned by the POP3 server. Emails can be retrieved or
	// deleted from the POP3 server via the UIDL. The header field for this property is
	// "X-UIDL".
	// 
	// Important: Emails downloaded via the IMAP protocol do not have UIDL's. UIDL's
	// are specific to the POP3 protocol. IMAP servers use UID's (notice the spelling
	// difference -- "UIDL" vs. "UID"). An email downloaded via the Chilkat IMAP
	// component will contain a "ckx-imap-uid" header field that holds either an IMAP
	// UID or sequence number. If the email was downloaded (from IMAP) using a UID,then
	// ckx-imap-uid contains a UID. If the email was downloaded by sequence number,
	// then it contains a sequence number (not unique and can change with a new
	// session). The "ckx-imap-isUid" header, which can have the value "YES" or "NO",
	// tells what ckx-imap-uid contains.
	// 
	const char *uidl(void);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	// 
	// As of v9.5.0.79, the only possible value is:
	//     "NO_FORMAT_FLOWED" - Don't automatically add "format=flowed" to any
	//     Content-Type header.
	// 
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	// 
	// As of v9.5.0.79, the only possible value is:
	//     "NO_FORMAT_FLOWED" - Don't automatically add "format=flowed" to any
	//     Content-Type header.
	// 
	const char *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	// 
	// As of v9.5.0.79, the only possible value is:
	//     "NO_FORMAT_FLOWED" - Don't automatically add "format=flowed" to any
	//     Content-Type header.
	// 
	void put_UncommonOptions(const char *newVal);

	// Applies to the UnpackHtml method. If true, then relative paths are used within
	// the HTML for the links to the related files (images and style sheets) that were
	// unpacked to the filesystem. Otherwise absolute paths are used. The default value
	// is true.
	bool get_UnpackUseRelPaths(void);
	// Applies to the UnpackHtml method. If true, then relative paths are used within
	// the HTML for the links to the related files (images and style sheets) that were
	// unpacked to the filesystem. Otherwise absolute paths are used. The default value
	// is true.
	void put_UnpackUseRelPaths(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds an attachment using the contents of a BinData object. If contentType is empty,
	// then the content-type will be inferred from the filename extension.
	bool AddAttachmentBd(const char *filename, CkBinData &binData, const char *contentType);


	// Adds or replaces a MIME header field in one of the email attachments. If the
	// header field does not exist, it is added. Otherwise it is replaced.
	void AddAttachmentHeader(int index, const char *fieldName, const char *fieldValue);


	// Adds a recipient to the blind carbon-copy list. emailAddress is required, but friendlyName may
	// be empty.
	// 
	// To add multiple BCC recipients, call this method once per recipient.
	// 
	bool AddBcc(const char *friendlyName, const char *emailAddress);


	// Adds a recipient to the carbon-copy list. emailAddress is required, but friendlyName may be
	// empty.
	// 
	// To add multiple CC recipients, call this method once per recipient.
	// 
	bool AddCC(const char *friendlyName, const char *emailAddress);


	// Adds an attachment directly from data in memory to the email.
	bool AddDataAttachment(const char *fileName, CkByteData &content);


	// Adds an attachment to an email from in-memory data. Same as AddDataAttachment
	// but allows the content-type to be specified.
	bool AddDataAttachment2(const char *fileName, CkByteData &content, const char *contentType);


	// Allows for certificates to be explicitly specified for sending encrypted email
	// to one or more recipients. Call this method once per certificate to be used. The
	// ClearEncryptCerts method may be called to clear the list of explicitly-specified
	// certificates.
	// 
	// Note: It is possible to send encrypted email without explicitly specifying the
	// certificates. On Windows computers, the registry-based Current-User and
	// Local-Machine certificate stores are automatically searched for certs matching
	// each of the recipients (To, CC, and BCC recipients).
	// 
	// Note: The SetEncryptCert method is equivalent to calling ClearEncryptCerts
	// followed by AddEncryptCert.
	// 
	bool AddEncryptCert(CkCert &cert);


	// Adds a file as an attachment to the email. Returns the MIME content-type of the
	// attachment, which is inferred based on the filename extension.
	bool AddFileAttachment(const char *path, CkString &outStrContentType);

	// Adds a file as an attachment to the email. Returns the MIME content-type of the
	// attachment, which is inferred based on the filename extension.
	const char *addFileAttachment(const char *path);

	// Same as AddFileAttachment, but the content type can be explicitly specified.
	bool AddFileAttachment2(const char *path, const char *contentType);


	// Any standard or non-standard (custom) header field can be added to the email
	// with this method. One interesting feature is that all header fields whose name
	// begins with "CKX-" will not be included in the header when an email is sent.
	// These fields will be included when saved to or loaded from XML. This makes it
	// easy to include persistent meta-data with an email which your programs can use
	// in any way it chooses.
	// 
	// Important: This method will replace an already-existing header field. To allow
	// for adding duplicate header fields, call AddHeaderField2 (see below).
	// 
	void AddHeaderField(const char *fieldName, const char *fieldValue);


	// This method is the same as AddHeaderField, except that if the header field
	// already exists, it is not replaced. A duplicate header will be added.
	void AddHeaderField2(const char *fieldName, const char *fieldValue);


	// Sets the HTML body of the email. Use this method if there will be multiple
	// versions of the body, but in different formats, such as HTML and plain text.
	// Otherwise, set the body by calling the SetHtmlBody method.
	bool AddHtmlAlternativeBody(const char *body);


	// Adds an iCalendar (text/calendar) alternative body to the email. The icalContent
	// contains the content of the iCalendar data. A sample is shown here:
	// BEGIN:VCALENDAR
	// VERSION:2.0
	// PRODID:-//hacksw/handcal//NONSGML v1.0//EN
	// BEGIN:VEVENT
	// UID:uid1@example.com
	// DTSTAMP:19970714T170000Z
	// ORGANIZER;CN=John Doe:MAILTO:john.doe@example.com
	// DTSTART:19970714T170000Z
	// DTEND:19970715T035959Z
	// SUMMARY:Bastille Day Party
	// END:VEVENT
	// END:VCALENDAR
	// The methodName is the "method" attribute used in the Content-Type header field in the
	// alternative body. For example, if set to "REQUEST", then the alternative body's
	// header would look like this:
	// Content-Type: text/calendar; method=REQUEST
	// Content-Transfer-Encoding: base64
	bool AddiCalendarAlternativeBody(const char *icalContent, const char *methodName);


	// Adds multiple recipients to the blind carbon-copy list. The parameter is a
	// string containing a comma separated list of full email addresses. Returns True
	// if successful.
	bool AddMultipleBcc(const char *commaSeparatedAddresses);


	// Adds multiple recipients to the carbon-copy list. The parameter is a string
	// containing a comma separated list of full email addresses. Returns True if
	// successful.
	bool AddMultipleCC(const char *commaSeparatedAddresses);


	// Adds multiple recipients to the "to" list. The parameter is a string containing
	// a comma separated list of full email addresses. Returns True if successful.
	bool AddMultipleTo(const char *commaSeparatedAddresses);


	// Adds a PFX to the object's internal list of sources to be searched for
	// certificates and private keys when decrypting. Multiple PFX sources can be added
	// by calling this method once for each. (On the Windows operating system, the
	// registry-based certificate stores are also automatically searched, so it is
	// commonly not required to explicitly add PFX sources.)
	// 
	// The pfxBytes contains the bytes of a PFX file (also known as PKCS12 or .p12).
	// 
	bool AddPfxSourceData(CkByteData &pfxBytes, const char *pfxPassword);


	// Adds a PFX file to the object's internal list of sources to be searched for
	// certificates and private keys when decrypting. Multiple PFX files can be added
	// by calling this method once for each. (On the Windows operating system, the
	// registry-based certificate stores are also automatically searched, so it is
	// commonly not required to explicitly add PFX sources.)
	// 
	// The pfxFilePath contains the bytes of a PFX file (also known as PKCS12 or .p12).
	// 
	bool AddPfxSourceFile(const char *pfxFilePath, const char *pfxPassword);


	// Sets the plain-text body of the email. Use this method if there will be multiple
	// versions of the body, but in different formats, such as HTML and plain text.
	// Otherwise, simply set the Body property.
	bool AddPlainTextAlternativeBody(const char *body);


	// Adds a related item using the contents of a BinData object. Returns the
	// Content-ID for the newly added relted item.
	bool AddRelatedBd(const char *filename, CkBinData &binData, CkString &outStr);

	// Adds a related item using the contents of a BinData object. Returns the
	// Content-ID for the newly added relted item.
	const char *addRelatedBd(const char *filename, CkBinData &binData);

	// Adds a related item using the contents of a BinData object. The fileNameInHtml should be
	// set to the filename/path/url used in the corresponding HTML IMG tag's "src"
	// attribute.
	bool AddRelatedBd2(CkBinData &binData, const char *fileNameInHtml);


	// Adds the memory data as a related item to the email and returns the Content-ID.
	// Emails formatted in HTML can include images with this call and internally
	// reference the image through a "cid"hyperlink. (Chilkat Email.NET fully supports
	// the MHTML standard.)
	bool AddRelatedData(const char *fileName, CkByteData &inData, CkString &outStr);

	// Adds the memory data as a related item to the email and returns the Content-ID.
	// Emails formatted in HTML can include images with this call and internally
	// reference the image through a "cid"hyperlink. (Chilkat Email.NET fully supports
	// the MHTML standard.)
	const char *addRelatedData(const char *fileName, CkByteData &inData);

	// Adds a related item to the email from in-memory byte data. Related items are
	// things such as images and style sheets that are embedded within an HTML email.
	// They are not considered attachments because their sole purpose is to participate
	// in the display of the HTML. This method differs from AddRelatedData in that it
	// does not use or return a Content-ID. The filename argument should be set to the
	// filename used in the HTML img tag's src attribute (if it's an image), or the URL
	// referenced in an HTML link tag for a stylesheet.
	void AddRelatedData2(CkByteData &inData, const char *fileNameInHtml);


#if !defined(CHILKAT_MONO)
	// The same as AddRelatedData2, except the data is passed in as a "const unsigned
	// char *" with the byte count in szBytes.
	void AddRelatedData2P(const void *pBytes, unsigned long szBytes, const char *fileNameInHtml);

#endif

#if !defined(CHILKAT_MONO)
	// The same as AddRelatedData, except the data is passed in as a "const unsigned
	// char *" with the byte count in szBytes. The Content-ID assigned to the related item
	// is returned (in ARG4 for the upper-case alternative for this method).
	bool AddRelatedDataP(const char *nameInHtml, const void *pBytes, unsigned long szBytes, CkString &outStrContentId);

	// The same as AddRelatedData, except the data is passed in as a "const unsigned
	// char *" with the byte count in szBytes. The Content-ID assigned to the related item
	// is returned (in ARG4 for the upper-case alternative for this method).
	const char *addRelatedDataP(const char *nameInHtml, const void *pBytes, unsigned long szBytes);
#endif

	// Adds the contents of a file to the email and returns the Content-ID. Emails
	// formatted in HTML can include images with this call and internally reference the
	// image through a "cid" hyperlink. (Chilkat Email.NET fully supports the MHTML
	// standard.)
	bool AddRelatedFile(const char *path, CkString &outStrContentID);

	// Adds the contents of a file to the email and returns the Content-ID. Emails
	// formatted in HTML can include images with this call and internally reference the
	// image through a "cid" hyperlink. (Chilkat Email.NET fully supports the MHTML
	// standard.)
	const char *addRelatedFile(const char *path);

	// Adds a related item to the email from a file. Related items are things such as
	// images and style sheets that are embedded within an HTML email. They are not
	// considered attachments because their sole purpose is to participate in the
	// display of the HTML. This method differs from AddRelatedFile in that it does not
	// use or return a Content-ID. The filenameInHtml argument should be set to the filename used
	// in the HTML img tag's src attribute (if it's an image), or the URL referenced in
	// an HTML link tag for a stylesheet. The filenameOnDisk is the path in the local filesystem
	// of the file to be added.
	// 
	// Note: Outlook.com will not properly display embedded HTMl images when the filenameInHtml
	// includes a path part. Apparently, Outlook.com is only capable of correctly
	// displaying images when the filenameInHtml is a only a filename. Other email clients, such
	// as Mozilla Thunderbird, have no trouble when the filenameInHtml includes a path part.
	// 
	bool AddRelatedFile2(const char *filenameOnDisk, const char *filenameInHtml);


	// Adds or replaces a MIME header field in one of the email's related items. If the
	// header field does not exist, it is added. Otherwise it is replaced.
	void AddRelatedHeader(int index, const char *fieldName, const char *fieldValue);


	// Adds a related item to the email. A related item is typically an image or style
	// sheet referenced by an HTML tag within the HTML email body. The contents of the
	// related item are passed str. nameInHtml specifies the filename that should be used
	// within the HTML, and not an actual filename on the local filesystem. charset
	// specifies the charset that should be used for the text content of the related
	// item. Returns the content-ID generated for the added item.
	bool AddRelatedString(const char *nameInHtml, const char *str, const char *charset, CkString &outCid);

	// Adds a related item to the email. A related item is typically an image or style
	// sheet referenced by an HTML tag within the HTML email body. The contents of the
	// related item are passed str. nameInHtml specifies the filename that should be used
	// within the HTML, and not an actual filename on the local filesystem. charset
	// specifies the charset that should be used for the text content of the related
	// item. Returns the content-ID generated for the added item.
	const char *addRelatedString(const char *nameInHtml, const char *str, const char *charset);

	// Adds a related item to the email from an in-memory string. Related items are
	// things such as images and style sheets that are embedded within an HTML email.
	// They are not considered attachments because their sole purpose is to participate
	// in the display of the HTML. The filenameInHtml argument should be set to the
	// filename used in the HTML img tag's src attribute (if it's an image), or the URL
	// referenced in an HTML link tag for a stylesheet. The charset argument indicates
	// that the content should first be converted to the specified charset prior to
	// adding to the email. It should hava a value such as "iso-8859-1", "utf-8",
	// "Shift_JIS", etc.
	void AddRelatedString2(const char *fileNameInHtml, const char *content, const char *charset);


	// Adds an attachment directly from a string in memory to the email.
	bool AddStringAttachment(const char *path, const char *content);


	// Adds an attachment to an email. The path specifies the filename to be used for
	// the attachment and is not an actual filename existing on the local filesystem.
	// The content contains the text data for the attachment. The string will be converted
	// to the charset specified in charset before being added to the email.
	// 
	// Note: Beginning in v9.5.0.48, the charset may be prepended with "bom-" or "no-bom-"
	// to include or exclude the BOM (preamble) for charsets such as utf-16 or utf-8.
	// For example: "no-bom-utf-8" or "bom-utf-8".
	// 
	bool AddStringAttachment2(const char *path, const char *content, const char *charset);


	// Adds a recipient to the "to" list. emailAddress is required, but friendlyName may be empty.
	// Emails that have no "To" recipients will be sent to
	// _LT_undisclosed-recipients_GT_.
	// 
	// To add multiple recipients, call this method once per TO recipient.
	// 
	bool AddTo(const char *friendlyName, const char *emailAddress);


	// Decrypts and restores an email message that was previously encrypted using
	// AesEncrypt. The password must match the password used for encryption.
	bool AesDecrypt(const char *password);


	// Encrypts the email body, all alternative bodies, all message sub-parts and
	// attachments using 128-bit AES CBC encryption. Decrypting is achieved by calling
	// AesDecrypt with the same password. The AesEncrypt/Decrypt methods use symmetric
	// password-based AES encryption and greatly simplify sending and receiving
	// encrypted emails because certificates and private keys are not used. However,
	// the sending and receiving applications must both use Chilkat, and the password
	// must be pre-known on both ends.
	bool AesEncrypt(const char *password);


	// Appends a string to the plain-text body.
	void AppendToBody(const char *str);


	// Sometimes emails created by other software or systems are not formatted
	// according to typical conventions. This method provides a means to automatically
	// fix certain problems.
	// 
	// The fixups set to a comma-separated list of keywords that identify the fixups to
	// be applied. At the moment, there is only one fixup defined ("FixRelated") as
	// described here:
	//     FixRelated: Fixes the email so that HTML related items (images for example)
	//     are properly located in the email MIME structure. This prevents them from being
	//     seen as attachments.
	// 
	bool ApplyFixups(const char *fixups);


	// Please see the examples at the following pages for detailed information:
	bool AspUnpack(const char *prefix, const char *saveDir, const char *urlPath, bool cleanFiles);


	// Please see the examples at the following pages for detailed information:
	bool AspUnpack2(const char *prefix, const char *saveDir, const char *urlPath, bool cleanFiles, CkByteData &outHtml);


	// Attaches copy of the email to this email object. The attached email will be
	// encapsulated in an message/rfc822 sub-part.
	bool AttachEmail(CkEmail &email);


	// Attaches a MIME message to the email object. The attached MIME will be
	// encapsulated in an message/rfc822 sub-part.
	bool AttachMessage(CkByteData &mimeBytes);


	// Takes a byte array of multibyte (non-Unicode) data and returns a Unicode
	// B-Encoded string.
	bool BEncodeBytes(CkByteData &inData, const char *charset, CkString &outEncodedStr);

	// Takes a byte array of multibyte (non-Unicode) data and returns a Unicode
	// B-Encoded string.
	const char *bEncodeBytes(CkByteData &inData, const char *charset);

	// Takes a Unicode string, converts it to the charset specified in the 2nd
	// parameter, B-Encodes the converted multibyte data, and returns the encoded
	// Unicode string.
	bool BEncodeString(const char *str, const char *charset, CkString &outEncodedStr);

	// Takes a Unicode string, converts it to the charset specified in the 2nd
	// parameter, B-Encodes the converted multibyte data, and returns the encoded
	// Unicode string.
	const char *bEncodeString(const char *str, const char *charset);

	// Clears the email object of all information to the state as if the object was
	// just created, which will have default headers such as Mime-Version, Date,
	// Message-ID, Content-Type, Content-Transfer-Encoding, and X-Priority.
	void Clear(void);


	// Clears the list of blind carbon-copy recipients.
	void ClearBcc(void);


	// Clears the list of carbon-copy recipients.
	void ClearCC(void);


	// Clears the internal list of explicitly specified certificates to be used for
	// this encrypted email.
	void ClearEncryptCerts(void);


	// Clears the list of "to" recipients.
	void ClearTo(void);


	// Creates and returns an identical copy of the Email object.
	// The caller is responsible for deleting the object returned by this method.
	CkEmail *Clone(void);


	// Important: New programs should ComputeGlobalKey2 instead. This method did not
	// adequately canonicalize the string passed to the digest-MD5 hash and therefore
	// different versions of Chilkat may produce different results with this method.
	// 
	// Computes a global unique key for the email. The key is created by a digest-MD5
	// hash of the concatenation of the following header fields: Message-ID, Subject,
	// From, Date, To. (The header fields are Q/B decoded if necessary, converted to
	// the utf-8 encoding, concatenated, and hashed using MD5.) The 16-byte MD5 hash is
	// returned as an encoded string. The encoding determines the encoding: base64, hex,
	// url, etc. If bFold is true, then the 16-byte MD5 hash is folded to 8 bytes with
	// an XOR to produce a shorter key.
	// 
	bool ComputeGlobalKey(const char *encoding, bool bFold, CkString &outStr);

	// Important: New programs should ComputeGlobalKey2 instead. This method did not
	// adequately canonicalize the string passed to the digest-MD5 hash and therefore
	// different versions of Chilkat may produce different results with this method.
	// 
	// Computes a global unique key for the email. The key is created by a digest-MD5
	// hash of the concatenation of the following header fields: Message-ID, Subject,
	// From, Date, To. (The header fields are Q/B decoded if necessary, converted to
	// the utf-8 encoding, concatenated, and hashed using MD5.) The 16-byte MD5 hash is
	// returned as an encoded string. The encoding determines the encoding: base64, hex,
	// url, etc. If bFold is true, then the 16-byte MD5 hash is folded to 8 bytes with
	// an XOR to produce a shorter key.
	// 
	const char *computeGlobalKey(const char *encoding, bool bFold);

	// Computes a global unique key for the email. The key is created by a digest-MD5
	// hash of the concatenation of the following:
	// messageID + CRLF + subject + CRLF + from + CRLF + date + CRLF + recipientAddrs
	// 
	// messageID contains the contents of the Message-ID header field.
	// subject contains the contents of the Subject header field, trimmed of whitespace from both ends, 
	//     where TAB chars are converted to SPACE chars, and internal whitespace is trimmed so that 
	//    no more than one SPACE char in a row exists.
	// from contains the lowercase FROM header email address.
	// date contains the contents of the DATE header field.
	// toAddrs contains lowercase TO and CC recipient email addresses, comma separated, with duplicates removed, and sorted 
	//     in ascending order.  The BCC addresses are NOT included.
	// 
	// (After calling this method, the LastErrorText property can be examined to see the string that was hashed.)
	// The 16-byte MD5 hash is returned as an encoded string. The encoding determines the
	// encoding: base64, hex, url, etc. If bFold is true, then the 16-byte MD5 hash is
	// folded to 8 bytes with an XOR to produce a shorter key.
	bool ComputeGlobalKey2(const char *encoding, bool bFold, CkString &outStr);

	// Computes a global unique key for the email. The key is created by a digest-MD5
	// hash of the concatenation of the following:
	// messageID + CRLF + subject + CRLF + from + CRLF + date + CRLF + recipientAddrs
	// 
	// messageID contains the contents of the Message-ID header field.
	// subject contains the contents of the Subject header field, trimmed of whitespace from both ends, 
	//     where TAB chars are converted to SPACE chars, and internal whitespace is trimmed so that 
	//    no more than one SPACE char in a row exists.
	// from contains the lowercase FROM header email address.
	// date contains the contents of the DATE header field.
	// toAddrs contains lowercase TO and CC recipient email addresses, comma separated, with duplicates removed, and sorted 
	//     in ascending order.  The BCC addresses are NOT included.
	// 
	// (After calling this method, the LastErrorText property can be examined to see the string that was hashed.)
	// The 16-byte MD5 hash is returned as an encoded string. The encoding determines the
	// encoding: base64, hex, url, etc. If bFold is true, then the 16-byte MD5 hash is
	// folded to 8 bytes with an XOR to produce a shorter key.
	const char *computeGlobalKey2(const char *encoding, bool bFold);

	// Converts images embedded inline within HTML to multipart/related MIME parts
	// referenced from the HTML by CID.
	bool ConvertInlineImages(void);


	// Creates a new DSN (Delivery Status Notification) email having the format as
	// specified in RFC 3464. See the example (below) for more detailed information.
	// The caller is responsible for deleting the object returned by this method.
	CkEmail *CreateDsn(const char *humanReadableMessage, const char *xmlStatusFields, bool bHeaderOnly);


	// Returns a copy of the Email object with the body and header fields changed so
	// that the newly created email can be forwarded. After calling CreateForward,
	// simply add new recipients to the created email, and call MailMan.SendEmail.
	// The caller is responsible for deleting the object returned by this method.
	CkEmail *CreateForward(void);


	// Creates a new MDN (Message Disposition Notification) email having the format as
	// specified in RFC 3798. See the example (below) for more detailed information.
	// The caller is responsible for deleting the object returned by this method.
	CkEmail *CreateMdn(const char *humanReadableMessage, const char *xmlStatusFields, bool bHeaderOnly);


	// Returns a copy of the Email object with the body and header fields changed so
	// that the newly created email can be sent as a reply. After calling CreateReply,
	// simply prepend additional information to the body, and call MailMan.SendEmail.
	// 
	// Note: Attachments are not included in the returned reply email. However,
	// attached messages are included. If an application does not wish to include the
	// attached messages in a reply email, they can be removed by calling
	// RemoveAttachedMessages on the reply email object.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkEmail *CreateReply(void);


	// Saves the email to a temporary MHT file so that a WebBrowser control can
	// navigate to it and display it. If fileName is empty, a temporary filename is
	// generated and returned. If fileName is non-empty, then it will be created or
	// overwritten, and the input filename is simply returned.The MHT file that is
	// created will not contain any of the email's attachments, if any existed. Also,
	// if the email was plain-text, the MHT file will be saved such that the plain-text
	// is converted to HTML using pre-formatted text ("pre" HTML tags) allowing it to
	// be displayed correctly in a WebBrowser.
	bool CreateTempMht(const char *inFilename, CkString &outPath);

	// Saves the email to a temporary MHT file so that a WebBrowser control can
	// navigate to it and display it. If fileName is empty, a temporary filename is
	// generated and returned. If fileName is non-empty, then it will be created or
	// overwritten, and the input filename is simply returned.The MHT file that is
	// created will not contain any of the email's attachments, if any existed. Also,
	// if the email was plain-text, the MHT file will be saved such that the plain-text
	// is converted to HTML using pre-formatted text ("pre" HTML tags) allowing it to
	// be displayed correctly in a WebBrowser.
	const char *createTempMht(const char *inFilename);

	// Removes all attachments from the email.
	void DropAttachments(void);


	// A related item is typically an embedded image referenced from the HTML in the
	// email via a "CID" hyperlink. This method removes the Nth embedded image from the
	// email. Note: If the HTML tries to reference the removed image, it will be
	// displayed as a broken image link.
	void DropRelatedItem(int index);


	// A related item is typically an embedded image referenced from the HTML in the
	// email via a "CID" hyperlink. This method removes all the embedded images from
	// the email.
	void DropRelatedItems(void);


	// Drops a single attachment from the email. Returns True if successful.
	bool DropSingleAttachment(int index);


	// Digitally signed and/or encrypted emails are automatically "unwrapped" when
	// received from a POP3 or IMAP server, or when loaded from any source such as a
	// MIME string, in-memory byte data, or a .eml file. The results of the signature
	// verification / decryption are stored in the properties such as ReceivedSigned,
	// ReceivedEncrypted, SignaturesValid, etc. The signing certificate can be obtained
	// via the GetSigningCert function. If the signature contained more certificates in
	// the chain of authentication, this method provides a means to access them.
	// 
	// During signature verification, the email object collects the certs found in the
	// signature and holds onto them internally. To get the issuing certificate of the
	// signing certificate, call this method passing the cert returned by
	// GetSigningCert. If the issuing cert is available, it is returned. Otherwise
	// _NULL_ is returned. If the cert passed in is the root (i.e. a self-signed
	// certificate), then the cert object returned is a copy of the cert passed in.
	// 
	// To traverse the chain to the root, one would write a loop that on first
	// iteration passes the cert returned by GetSignedByCert (not GetSignerCert), and
	// then on each subsequent iteration passes the cert from the previous iteration.
	// The loop would exit when a cert is returned that has the same SubjectDN and
	// SerialNumber as what was passed in (or when FindIssuer returns _NULL_).
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCert *FindIssuer(CkCert &cert);


	// Generates a unique filename for this email. The filename will be different each
	// time the method is called.
	bool GenerateFilename(CkString &outStrFilename);

	// Generates a unique filename for this email. The filename will be different each
	// time the method is called.
	const char *generateFilename(void);

	// Returns the Nth alternative body. The NumAlternatives property tells the number
	// of alternative bodies present. Use the GetHtmlBody and GetPlainTextBody methods
	// to easily get the HTML or plain text alternative bodies.
	bool GetAlternativeBody(int index, CkString &outStrBody);

	// Returns the Nth alternative body. The NumAlternatives property tells the number
	// of alternative bodies present. Use the GetHtmlBody and GetPlainTextBody methods
	// to easily get the HTML or plain text alternative bodies.
	const char *getAlternativeBody(int index);
	// Returns the Nth alternative body. The NumAlternatives property tells the number
	// of alternative bodies present. Use the GetHtmlBody and GetPlainTextBody methods
	// to easily get the HTML or plain text alternative bodies.
	const char *alternativeBody(int index);


	// Returns contents of the Nth alternative body to binData. The 1st alternative body
	// is at index 0. This method should only be called when the NumAlternatives
	// property has a value greater than 0.
	bool GetAlternativeBodyBd(int index, CkBinData &binData);


	// Returns the alternative body by content-type, such as "text/plain", "text/html",
	// "text/xml", etc.
	bool GetAlternativeBodyByContentType(const char *contentType, CkString &outStr);

	// Returns the alternative body by content-type, such as "text/plain", "text/html",
	// "text/xml", etc.
	const char *getAlternativeBodyByContentType(const char *contentType);
	// Returns the alternative body by content-type, such as "text/plain", "text/html",
	// "text/xml", etc.
	const char *alternativeBodyByContentType(const char *contentType);


	// Returns the content type of the Nth alternative body. The NumAlternatives
	// property tells the number of alternative bodies present.
	bool GetAlternativeContentType(int index, CkString &outStrContentType);

	// Returns the content type of the Nth alternative body. The NumAlternatives
	// property tells the number of alternative bodies present.
	const char *getAlternativeContentType(int index);
	// Returns the content type of the Nth alternative body. The NumAlternatives
	// property tells the number of alternative bodies present.
	const char *alternativeContentType(int index);


	// Returns the value of a header field within the Nth alternative body's MIME
	// sub-part.
	bool GetAltHeaderField(int index, const char *fieldName, CkString &outStr);

	// Returns the value of a header field within the Nth alternative body's MIME
	// sub-part.
	const char *getAltHeaderField(int index, const char *fieldName);
	// Returns the value of a header field within the Nth alternative body's MIME
	// sub-part.
	const char *altHeaderField(int index, const char *fieldName);


	// Returns an embedded "message/rfc822" subpart as an email object. (Emails are
	// embedded as "message/rfc822" subparts by some mail clients when forwarding an
	// email.) This method allows the original email to be accessed.
	// The caller is responsible for deleting the object returned by this method.
	CkEmail *GetAttachedMessage(int index);


	// Returns a header field attribute value for the Nth attached (embedded) email.
	// For example, to get the value of the "name" attribute in the Content-Type header
	// for the 1st attached message:
	// Content-Type: message/rfc822; name="md75000024149.eml"
	// then the method arguments should contain the values 0, "Content-Type", "name".
	bool GetAttachedMessageAttr(int index, const char *fieldName, const char *attrName, CkString &outStr);

	// Returns a header field attribute value for the Nth attached (embedded) email.
	// For example, to get the value of the "name" attribute in the Content-Type header
	// for the 1st attached message:
	// Content-Type: message/rfc822; name="md75000024149.eml"
	// then the method arguments should contain the values 0, "Content-Type", "name".
	const char *getAttachedMessageAttr(int index, const char *fieldName, const char *attrName);
	// Returns a header field attribute value for the Nth attached (embedded) email.
	// For example, to get the value of the "name" attribute in the Content-Type header
	// for the 1st attached message:
	// Content-Type: message/rfc822; name="md75000024149.eml"
	// then the method arguments should contain the values 0, "Content-Type", "name".
	const char *attachedMessageAttr(int index, const char *fieldName, const char *attrName);


	// Returns the filename of the Nth attached (embedded) email. The filename is the
	// "filename" attribute of the content-disposition header field found within the
	// Nth message/rfc822 sub-part of the calling email object.
	// 
	// Important: The attached message filename is only present if the
	// Content-Disposition header exists AND contains a "filename" attribute. If
	// questions arise, one could open the email in a text editor to examine the MIME
	// sub-header for the attached message (where the Content-Type = "message/rfc822").
	// For example, here is a sub-header that has a filename:
	// Content-Type: message/rfc822; name="GetAttachedMessageAttr.eml"
	// Content-Transfer-Encoding: 7bit
	// Content-Disposition: attachment; filename="GetAttachedMessageAttr.eml"
	// Here is an attached message sub-header that does NOT have a filename:
	// Content-Type: message/rfc822
	// Content-Transfer-Encoding: 7bit
	// Content-Disposition: attachment
	// 
	bool GetAttachedMessageFilename(int index, CkString &outStr);

	// Returns the filename of the Nth attached (embedded) email. The filename is the
	// "filename" attribute of the content-disposition header field found within the
	// Nth message/rfc822 sub-part of the calling email object.
	// 
	// Important: The attached message filename is only present if the
	// Content-Disposition header exists AND contains a "filename" attribute. If
	// questions arise, one could open the email in a text editor to examine the MIME
	// sub-header for the attached message (where the Content-Type = "message/rfc822").
	// For example, here is a sub-header that has a filename:
	// Content-Type: message/rfc822; name="GetAttachedMessageAttr.eml"
	// Content-Transfer-Encoding: 7bit
	// Content-Disposition: attachment; filename="GetAttachedMessageAttr.eml"
	// Here is an attached message sub-header that does NOT have a filename:
	// Content-Type: message/rfc822
	// Content-Transfer-Encoding: 7bit
	// Content-Disposition: attachment
	// 
	const char *getAttachedMessageFilename(int index);
	// Returns the filename of the Nth attached (embedded) email. The filename is the
	// "filename" attribute of the content-disposition header field found within the
	// Nth message/rfc822 sub-part of the calling email object.
	// 
	// Important: The attached message filename is only present if the
	// Content-Disposition header exists AND contains a "filename" attribute. If
	// questions arise, one could open the email in a text editor to examine the MIME
	// sub-header for the attached message (where the Content-Type = "message/rfc822").
	// For example, here is a sub-header that has a filename:
	// Content-Type: message/rfc822; name="GetAttachedMessageAttr.eml"
	// Content-Transfer-Encoding: 7bit
	// Content-Disposition: attachment; filename="GetAttachedMessageAttr.eml"
	// Here is an attached message sub-header that does NOT have a filename:
	// Content-Type: message/rfc822
	// Content-Transfer-Encoding: 7bit
	// Content-Disposition: attachment
	// 
	const char *attachedMessageFilename(int index);


	// Returns a header field attribute value from the header field of the Nth
	// attachment.
	bool GetAttachmentAttr(int index, const char *fieldName, const char *attrName, CkString &outStr);

	// Returns a header field attribute value from the header field of the Nth
	// attachment.
	const char *getAttachmentAttr(int index, const char *fieldName, const char *attrName);
	// Returns a header field attribute value from the header field of the Nth
	// attachment.
	const char *attachmentAttr(int index, const char *fieldName, const char *attrName);


	// Copies an attachment's binary data into binData. The first attachment is at index
	// 0.
	bool GetAttachmentBd(int index, CkBinData &binData);


	// Returns the ContentID header field for the Nth attachment. The first attachment
	// is at index 0.
	bool GetAttachmentContentID(int index, CkString &outStrContentID);

	// Returns the ContentID header field for the Nth attachment. The first attachment
	// is at index 0.
	const char *getAttachmentContentID(int index);
	// Returns the ContentID header field for the Nth attachment. The first attachment
	// is at index 0.
	const char *attachmentContentID(int index);


	// Returns the Content-Type header field for the Nth attachment. Indexing of
	// attachments begins at 0.
	bool GetAttachmentContentType(int index, CkString &outStrContentType);

	// Returns the Content-Type header field for the Nth attachment. Indexing of
	// attachments begins at 0.
	const char *getAttachmentContentType(int index);
	// Returns the Content-Type header field for the Nth attachment. Indexing of
	// attachments begins at 0.
	const char *attachmentContentType(int index);


	// Retrieves an attachment's binary data for in-memory access.
	bool GetAttachmentData(int index, CkByteData &outData);


	// Retrieves an attachment's filename.
	bool GetAttachmentFilename(int index, CkString &outStrFilename);

	// Retrieves an attachment's filename.
	const char *getAttachmentFilename(int index);
	// Retrieves an attachment's filename.
	const char *attachmentFilename(int index);


	// Returns the value of a header field (by name) of an attachment.
	bool GetAttachmentHeader(int attachIndex, const char *fieldName, CkString &outFieldValue);

	// Returns the value of a header field (by name) of an attachment.
	const char *getAttachmentHeader(int attachIndex, const char *fieldName);
	// Returns the value of a header field (by name) of an attachment.
	const char *attachmentHeader(int attachIndex, const char *fieldName);


	// Returns the size (in bytes) of the Nth attachment. The 1st attachment is at
	// index 0. Returns -1 if there is no attachment at the specified index.
	int GetAttachmentSize(int index);


	// Retrieves an attachment's data as a String. All CRLF sequences will be
	// translated to single newline characters. The charset indicates how to interpret the
	// bytes of the attachment. For example, if the attachment is a text file using the
	// utf-8 byte representation for characters, pass "utf-8".
	bool GetAttachmentString(int index, const char *charset, CkString &outStr);

	// Retrieves an attachment's data as a String. All CRLF sequences will be
	// translated to single newline characters. The charset indicates how to interpret the
	// bytes of the attachment. For example, if the attachment is a text file using the
	// utf-8 byte representation for characters, pass "utf-8".
	const char *getAttachmentString(int index, const char *charset);
	// Retrieves an attachment's data as a String. All CRLF sequences will be
	// translated to single newline characters. The charset indicates how to interpret the
	// bytes of the attachment. For example, if the attachment is a text file using the
	// utf-8 byte representation for characters, pass "utf-8".
	const char *attachmentString(int index, const char *charset);


	// Retrieves an attachment's data as a String. All end-of-lines will be translated
	// to CRLF sequences.
	bool GetAttachmentStringCrLf(int index, const char *charset, CkString &outStrData);

	// Retrieves an attachment's data as a String. All end-of-lines will be translated
	// to CRLF sequences.
	const char *getAttachmentStringCrLf(int index, const char *charset);
	// Retrieves an attachment's data as a String. All end-of-lines will be translated
	// to CRLF sequences.
	const char *attachmentStringCrLf(int index, const char *charset);


	// Returns a blind carbon-copy recipient's full email address.
	bool GetBcc(int index, CkString &outStr);

	// Returns a blind carbon-copy recipient's full email address.
	const char *getBcc(int index);
	// Returns a blind carbon-copy recipient's full email address.
	const char *bcc(int index);


	// Returns the Nth BCC address (only the address part, not the friendly-name part).
	bool GetBccAddr(int index, CkString &outStr);

	// Returns the Nth BCC address (only the address part, not the friendly-name part).
	const char *getBccAddr(int index);
	// Returns the Nth BCC address (only the address part, not the friendly-name part).
	const char *bccAddr(int index);


	// Returns the Nth BCC name (only the friendly-name part, not the address part).
	bool GetBccName(int index, CkString &outStr);

	// Returns the Nth BCC name (only the friendly-name part, not the address part).
	const char *getBccName(int index);
	// Returns the Nth BCC name (only the friendly-name part, not the address part).
	const char *bccName(int index);


	// Returns a carbon-copy recipient's full email address.
	bool GetCC(int index, CkString &outStr);

	// Returns a carbon-copy recipient's full email address.
	const char *getCC(int index);
	// Returns a carbon-copy recipient's full email address.
	const char *cC(int index);


	// Returns the Nth CC address (only the address part, not the friendly-name part).
	bool GetCcAddr(int index, CkString &outStr);

	// Returns the Nth CC address (only the address part, not the friendly-name part).
	const char *getCcAddr(int index);
	// Returns the Nth CC address (only the address part, not the friendly-name part).
	const char *ccAddr(int index);


	// Returns the Nth CC name (only the friendly-name part, not the address part).
	bool GetCcName(int index, CkString &outStr);

	// Returns the Nth CC name (only the friendly-name part, not the address part).
	const char *getCcName(int index);
	// Returns the Nth CC name (only the friendly-name part, not the address part).
	const char *ccName(int index);


	// If the email is a multipart/report, then it is a delivery status notification.
	// This method can be used to get individual pieces of information from the
	// message/delivery-status part of the email. This method should only be called if
	// the IsMultipartReport method returns true.
	// 
	// The fieldName should be set a string such as "Final-Recipient", "Status", "Action",
	// "Reporting-MTA", etc.
	// Reporting-MTA: dns; XYZ.abc.nl
	// 
	// Final-recipient: RFC822; someEmailAddr@doesnotexist123.nl
	// Action: failed
	// Status: 5.4.4
	// X-Supplementary-Info: 
	// 
	bool GetDeliveryStatusInfo(const char *fieldName, CkString &outFieldValue);

	// If the email is a multipart/report, then it is a delivery status notification.
	// This method can be used to get individual pieces of information from the
	// message/delivery-status part of the email. This method should only be called if
	// the IsMultipartReport method returns true.
	// 
	// The fieldName should be set a string such as "Final-Recipient", "Status", "Action",
	// "Reporting-MTA", etc.
	// Reporting-MTA: dns; XYZ.abc.nl
	// 
	// Final-recipient: RFC822; someEmailAddr@doesnotexist123.nl
	// Action: failed
	// Status: 5.4.4
	// X-Supplementary-Info: 
	// 
	const char *getDeliveryStatusInfo(const char *fieldName);
	// If the email is a multipart/report, then it is a delivery status notification.
	// This method can be used to get individual pieces of information from the
	// message/delivery-status part of the email. This method should only be called if
	// the IsMultipartReport method returns true.
	// 
	// The fieldName should be set a string such as "Final-Recipient", "Status", "Action",
	// "Reporting-MTA", etc.
	// Reporting-MTA: dns; XYZ.abc.nl
	// 
	// Final-recipient: RFC822; someEmailAddr@doesnotexist123.nl
	// Action: failed
	// Status: 5.4.4
	// X-Supplementary-Info: 
	// 
	const char *deliveryStatusInfo(const char *fieldName);


	// Returns an digest contained within a multipart/digest as an email object. The
	// 1st digest is at index 0. Use the NumDigests property to get the number of
	// digests available.
	// 
	// Note: This example requires Chilkat v9.5.0.66 or greater.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkEmail *GetDigest(int index);


	// If the email is a multipart/report, then it is a delivery status notification.
	// This method can be used to get Final-Recipient values from the
	// message/delivery-status part of the email. This method should only be called if
	// the IsMultipartReport method returns true.
	// The caller is responsible for deleting the object returned by this method.
	CkStringArray *GetDsnFinalRecipients(void);


	// Returns the date/time found in the "Date" header field as a date/time object.
	// The caller is responsible for deleting the object returned by this method.
	CkDateTime *GetDt(void);


	// Returns the certificate that was previously set by SetEncryptCert.
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetEncryptCert(void);


	// Returns the certificate associated with a received encrypted email.
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetEncryptedByCert(void);


	// Reads a file and returns the contents as a String. This is here purely for
	// convenience.
	bool GetFileContent(const char *path, CkByteData &outData);


	// Returns the value of a header field.
	bool GetHeaderField(const char *fieldName, CkString &outStrFieldData);

	// Returns the value of a header field.
	const char *getHeaderField(const char *fieldName);
	// Returns the value of a header field.
	const char *headerField(const char *fieldName);


	// Return the name of the Nth header field. The NumHeaderFields() method can be
	// used to get the number of header fields. The GetHeaderField() method can be used
	// to get the value of the field given the field name.
	// 
	// The 1st header field is at index 0. (All Chilkat indexing is 0-based.)
	// 
	bool GetHeaderFieldName(int index, CkString &outStrFieldName);

	// Return the name of the Nth header field. The NumHeaderFields() method can be
	// used to get the number of header fields. The GetHeaderField() method can be used
	// to get the value of the field given the field name.
	// 
	// The 1st header field is at index 0. (All Chilkat indexing is 0-based.)
	// 
	const char *getHeaderFieldName(int index);
	// Return the name of the Nth header field. The NumHeaderFields() method can be
	// used to get the number of header fields. The GetHeaderField() method can be used
	// to get the value of the field given the field name.
	// 
	// The 1st header field is at index 0. (All Chilkat indexing is 0-based.)
	// 
	const char *headerFieldName(int index);


	// Returns the value of the Nth header field. (Indexing begins at 0) The number of
	// header fields can be obtained from the NumHeaderFields property.
	// 
	// The 1st header field is at index 0, the last header field is at index
	// NumHeaderFields-1. (All Chilkat indexing is 0-based.)
	// 
	bool GetHeaderFieldValue(int index, CkString &outStrFieldValue);

	// Returns the value of the Nth header field. (Indexing begins at 0) The number of
	// header fields can be obtained from the NumHeaderFields property.
	// 
	// The 1st header field is at index 0, the last header field is at index
	// NumHeaderFields-1. (All Chilkat indexing is 0-based.)
	// 
	const char *getHeaderFieldValue(int index);
	// Returns the value of the Nth header field. (Indexing begins at 0) The number of
	// header fields can be obtained from the NumHeaderFields property.
	// 
	// The 1st header field is at index 0, the last header field is at index
	// NumHeaderFields-1. (All Chilkat indexing is 0-based.)
	// 
	const char *headerFieldValue(int index);


	// Returns the body having the "text/html" content type.
	bool GetHtmlBody(CkString &outStrBody);

	// Returns the body having the "text/html" content type.
	const char *getHtmlBody(void);
	// Returns the body having the "text/html" content type.
	const char *htmlBody(void);


	// When email headers are downloaded from an IMAP server (using Chilkat IMAP), a
	// "ckx-imap-uid" header field is added. The content of this header is the UID or
	// sequence number of the email on the IMAP server. In addition, a "ckx-imap-isUid"
	// header field is added, and this will have the value YES or NO. If the value is
	// YES, then ckx-imap-uid contains a UID, if the value is NO, then ckx-imap-uid
	// contains the sequence number. This method returns the UID if ckx-imap-uid exists
	// and contains a UID, otherwise it returns -1.
	// 
	// An application that wishes to download the full email would use this UID and
	// then call the Chilkat IMAP object's FetchSingle or FetchSingleAsMime methods.
	// 
	// Note:If an email was downloaded from the IMAP server in a way such that the UID
	// is not received, then there will be no "ckx-imap-uid" header field and this
	// method would return -1. For example, if emails are downloaded by sequence
	// numbers via the Imap.FetchSequence method, then UIDs are not used and therefore
	// the email object will not contain this information.
	// 
	int GetImapUid(void);


	// Parses an HTML email and returns the set of domain names that occur in
	// hyperlinks within the HTML body.
	// The caller is responsible for deleting the object returned by this method.
	CkStringArray *GetLinkedDomains(void);


	// Returns a header field's data in a byte array. If the field was Q or B encoded,
	// this is automatically decoded, and the raw bytes of the field are returned. Call
	// GetHeaderField to retrieve the header field as a Unicode string.
	// 
	// The 1st header field is at index 0. (All Chilkat indexing is 0-based.)
	// 
	bool GetMbHeaderField(const char *fieldName, const char *charset, CkByteData &outBytes);


	// Returns the HTML body converted to a specified charset. If no HTML body exists,
	// the returned byte array is empty. The returned data will be such that not only
	// is the character data converted (if necessary) to the convertToCharset, but the
	// HTML is edited to add or modify the META tag that specifies the charset within
	// the HTML.
	bool GetMbHtmlBody(const char *charset, CkByteData &outData);


	// Returns the plain-text body converted to a specified charset. The return value
	// is a byte array containing multibyte character data.
	bool GetMbPlainTextBody(const char *charset, CkByteData &outData);


	// Return the email as MIME text containing the email header, body (or bodies),
	// related items (if any), and all attachments
	bool GetMime(CkString &outStrMime);

	// Return the email as MIME text containing the email header, body (or bodies),
	// related items (if any), and all attachments
	const char *getMime(void);
	// Return the email as MIME text containing the email header, body (or bodies),
	// related items (if any), and all attachments
	const char *mime(void);


	// Return the email as binary MIME containing the email header, body (or bodies),
	// related items (if any), and all attachments. The MIME is appended to the
	// existing contents (if any) of bindat.
	bool GetMimeBd(CkBinData &bindat);


	// Returns the full MIME of an email.
	bool GetMimeBinary(CkByteData &outBytes);


	// Return the email as MIME text containing the email header, body (or bodies),
	// related items (if any), and all attachments. The MIME is appended to the
	// existing contents (if any) of sb.
	bool GetMimeSb(CkStringBuilder &sb);


	// Returns the binary bytes of the Nth MIME sub-part having a specified content
	// type (such as "application/pdf". Indexing begins at 0. Call GetNumPartsOfType to
	// find out how many MIME sub-parts exist for any given content type. If inlineOnly is
	// true, then only MIME sub-parts having a content-disposition of "inline" are
	// included. If excludeAttachments is true, then MIME sub-parts having a content-disposition of
	// "attachment" are excluded.
	// 
	// Note: If the email was downloaded as header-only, it will not contain all the
	// parts of the full email. Also, if downloaded from IMAP excluding attachments,
	// those parts that are the attachments will (of course) be missing.
	// 
	bool GetNthBinaryPartOfType(int index, const char *contentType, bool inlineOnly, bool excludeAttachments, CkByteData &outBytes);


	// Returns the text of the Nth MIME sub-part having a specified content type (such
	// as "text/plain". Indexing begins at 0. Call GetNumPartsOfType to find out how
	// many MIME sub-parts exist for any given content type. If inlineOnly is true, then
	// only MIME sub-parts having a content-disposition of "inline" are included. If
	// excludeAttachments is true, then MIME sub-parts having a content-disposition of "attachment"
	// are excluded.
	// 
	// Note: If the email was downloaded as header-only, it will not contain all the
	// parts of the full email. Also, if downloaded from IMAP excluding attachments,
	// those parts that are the attachments will (of course) be missing.
	// 
	bool GetNthTextPartOfType(int index, const char *contentType, bool inlineOnly, bool excludeAttachments, CkString &outStr);

	// Returns the text of the Nth MIME sub-part having a specified content type (such
	// as "text/plain". Indexing begins at 0. Call GetNumPartsOfType to find out how
	// many MIME sub-parts exist for any given content type. If inlineOnly is true, then
	// only MIME sub-parts having a content-disposition of "inline" are included. If
	// excludeAttachments is true, then MIME sub-parts having a content-disposition of "attachment"
	// are excluded.
	// 
	// Note: If the email was downloaded as header-only, it will not contain all the
	// parts of the full email. Also, if downloaded from IMAP excluding attachments,
	// those parts that are the attachments will (of course) be missing.
	// 
	const char *getNthTextPartOfType(int index, const char *contentType, bool inlineOnly, bool excludeAttachments);
	// Returns the text of the Nth MIME sub-part having a specified content type (such
	// as "text/plain". Indexing begins at 0. Call GetNumPartsOfType to find out how
	// many MIME sub-parts exist for any given content type. If inlineOnly is true, then
	// only MIME sub-parts having a content-disposition of "inline" are included. If
	// excludeAttachments is true, then MIME sub-parts having a content-disposition of "attachment"
	// are excluded.
	// 
	// Note: If the email was downloaded as header-only, it will not contain all the
	// parts of the full email. Also, if downloaded from IMAP excluding attachments,
	// those parts that are the attachments will (of course) be missing.
	// 
	const char *nthTextPartOfType(int index, const char *contentType, bool inlineOnly, bool excludeAttachments);


	// Returns the number of MIME sub-parts within the email having a specified content
	// type (such as "text/plain"). If inlineOnly is true, then only MIME sub-parts having
	// a content-disposition of "inline" are included. If excludeAttachments is true, then MIME
	// sub-parts having a content-disposition of "attachment" are excluded.
	// 
	// Note: If the email was downloaded as header-only, it will not contain all the
	// parts of the full email. Also, if downloaded from IMAP excluding attachments,
	// those parts that are the attachments will (of course) be missing.
	// 
	int GetNumPartsOfType(const char *contentType, bool inlineOnly, bool excludeAttachments);


	// Returns the email body having the "text/plain" content type.
	bool GetPlainTextBody(CkString &outStrBody);

	// Returns the email body having the "text/plain" content type.
	const char *getPlainTextBody(void);
	// Returns the email body having the "text/plain" content type.
	const char *plainTextBody(void);


	// Returns a header field attribute value from the header field of the Nth related
	// item.
	bool GetRelatedAttr(int index, const char *fieldName, const char *attrName, CkString &outStr);

	// Returns a header field attribute value from the header field of the Nth related
	// item.
	const char *getRelatedAttr(int index, const char *fieldName, const char *attrName);
	// Returns a header field attribute value from the header field of the Nth related
	// item.
	const char *relatedAttr(int index, const char *fieldName, const char *attrName);


	// Returns the content ID of a related item contained with the email. Related items
	// are typically images and style-sheets embedded within HTML emails.
	bool GetRelatedContentID(int index, CkString &outStrContentID);

	// Returns the content ID of a related item contained with the email. Related items
	// are typically images and style-sheets embedded within HTML emails.
	const char *getRelatedContentID(int index);
	// Returns the content ID of a related item contained with the email. Related items
	// are typically images and style-sheets embedded within HTML emails.
	const char *relatedContentID(int index);


	// Returns the Content-Location of a related item contained with the email. Related
	// items are typically images and style-sheets embedded within HTML emails.
	bool GetRelatedContentLocation(int index, CkString &outStr);

	// Returns the Content-Location of a related item contained with the email. Related
	// items are typically images and style-sheets embedded within HTML emails.
	const char *getRelatedContentLocation(int index);
	// Returns the Content-Location of a related item contained with the email. Related
	// items are typically images and style-sheets embedded within HTML emails.
	const char *relatedContentLocation(int index);


	// Returns the content-type of the Nth related content item in an email message.
	bool GetRelatedContentType(int index, CkString &outStrContentType);

	// Returns the content-type of the Nth related content item in an email message.
	const char *getRelatedContentType(int index);
	// Returns the content-type of the Nth related content item in an email message.
	const char *relatedContentType(int index);


	// Returns the content of a related item contained with the email. Related items
	// are typically images and style-sheets embedded within HTML emails.
	bool GetRelatedData(int index, CkByteData &outBuffer);


	// Returns the filename of a related item contained with the email. Related items
	// are typically images and style-sheets embedded within HTML emails.
	bool GetRelatedFilename(int index, CkString &outStrFilename);

	// Returns the filename of a related item contained with the email. Related items
	// are typically images and style-sheets embedded within HTML emails.
	const char *getRelatedFilename(int index);
	// Returns the filename of a related item contained with the email. Related items
	// are typically images and style-sheets embedded within HTML emails.
	const char *relatedFilename(int index);


	// Returns the text with CR line-endings of a related item contained with the
	// email. Related items are typically images and style-sheets embedded within HTML
	// emails.
	bool GetRelatedString(int index, const char *charset, CkString &outStrData);

	// Returns the text with CR line-endings of a related item contained with the
	// email. Related items are typically images and style-sheets embedded within HTML
	// emails.
	const char *getRelatedString(int index, const char *charset);
	// Returns the text with CR line-endings of a related item contained with the
	// email. Related items are typically images and style-sheets embedded within HTML
	// emails.
	const char *relatedString(int index, const char *charset);


	// Returns the text with CRLF line-endings of a related item contained with the
	// email. Related items are typically images and style-sheets embedded within HTML
	// emails.
	bool GetRelatedStringCrLf(int index, const char *charset, CkString &outStr);

	// Returns the text with CRLF line-endings of a related item contained with the
	// email. Related items are typically images and style-sheets embedded within HTML
	// emails.
	const char *getRelatedStringCrLf(int index, const char *charset);
	// Returns the text with CRLF line-endings of a related item contained with the
	// email. Related items are typically images and style-sheets embedded within HTML
	// emails.
	const char *relatedStringCrLf(int index, const char *charset);


	// Returns a replacement pattern previously defined for mail-merge operations.
	bool GetReplacePattern(int index, CkString &outStrPattern);

	// Returns a replacement pattern previously defined for mail-merge operations.
	const char *getReplacePattern(int index);
	// Returns a replacement pattern previously defined for mail-merge operations.
	const char *replacePattern(int index);


	// Returns a replacement string for a previously defined pattern/replacement string
	// pair. (This is a mail-merge feature.)
	bool GetReplaceString(int index, CkString &outStr);

	// Returns a replacement string for a previously defined pattern/replacement string
	// pair. (This is a mail-merge feature.)
	const char *getReplaceString(int index);
	// Returns a replacement string for a previously defined pattern/replacement string
	// pair. (This is a mail-merge feature.)
	const char *replaceString(int index);


	// Returns a replacement string for a previously defined pattern/replacement string
	// pair. (This is a mail-merge feature.)
	bool GetReplaceString2(const char *pattern, CkString &outStr);

	// Returns a replacement string for a previously defined pattern/replacement string
	// pair. (This is a mail-merge feature.)
	const char *getReplaceString2(const char *pattern);
	// Returns a replacement string for a previously defined pattern/replacement string
	// pair. (This is a mail-merge feature.)
	const char *replaceString2(const char *pattern);


	// (See the NumReports property.) Returns the body content of the Nth report within
	// a multipart/report email.
	// 
	// Multipart/report is a message type that contains data formatted for a mail
	// server to read. It is split between a text/plain (or some other content/type
	// easily readable) and a message/delivery-status, which contains the data
	// formatted for the mail server to read.
	// 
	// It is defined in RFC 3462
	// 
	bool GetReport(int index, CkString &outStr);

	// (See the NumReports property.) Returns the body content of the Nth report within
	// a multipart/report email.
	// 
	// Multipart/report is a message type that contains data formatted for a mail
	// server to read. It is split between a text/plain (or some other content/type
	// easily readable) and a message/delivery-status, which contains the data
	// formatted for the mail server to read.
	// 
	// It is defined in RFC 3462
	// 
	const char *getReport(int index);
	// (See the NumReports property.) Returns the body content of the Nth report within
	// a multipart/report email.
	// 
	// Multipart/report is a message type that contains data formatted for a mail
	// server to read. It is split between a text/plain (or some other content/type
	// easily readable) and a message/delivery-status, which contains the data
	// formatted for the mail server to read.
	// 
	// It is defined in RFC 3462
	// 
	const char *report(int index);


	// Return the certificate used to digitally sign this email.
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetSignedByCert(void);


	// Return the full certificate chain of the certificate used to digitally sign this
	// email.
	// The caller is responsible for deleting the object returned by this method.
	CkCertChain *GetSignedByCertChain(void);


	// Return the certificate that will be used to digitally sign this email. This is
	// the cerficate that was previously set by calling the SetSigningCert method.
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetSigningCert(void);


	// Returns a "to" recipient's full email address.
	bool GetTo(int index, CkString &outStr);

	// Returns a "to" recipient's full email address.
	const char *getTo(int index);
	// Returns a "to" recipient's full email address.
	const char *to(int index);


	// Returns the Nth To address (only the address part, not the friendly-name part).
	bool GetToAddr(int index, CkString &outStr);

	// Returns the Nth To address (only the address part, not the friendly-name part).
	const char *getToAddr(int index);
	// Returns the Nth To address (only the address part, not the friendly-name part).
	const char *toAddr(int index);


	// Returns the Nth To name (only the friendly-name part, not the address part).
	bool GetToName(int index, CkString &outStr);

	// Returns the Nth To name (only the friendly-name part, not the address part).
	const char *getToName(int index);
	// Returns the Nth To name (only the friendly-name part, not the address part).
	const char *toName(int index);


	// Convert the email object to an XML document in memory
	bool GetXml(CkString &outStrXml);

	// Convert the email object to an XML document in memory
	const char *getXml(void);
	// Convert the email object to an XML document in memory
	const char *xml(void);


	// Returns true if the email has a header field with the specified fieldName with a
	// value matching valuePattern. Case sensitivity is controlled by caseSensitive. The valuePattern may
	// include 0 or more asterisk (wildcard) characters which match 0 or more of any
	// character.
	bool HasHeaderMatching(const char *fieldName, const char *valuePattern, bool caseSensitive);


	// Returns true if the email has an HTML body.
	bool HasHtmlBody(void);


	// Returns true if the email has a plain-text body.
	bool HasPlainTextBody(void);


	// Returns true if the email is a multipart/report email.
	bool IsMultipartReport(void);


	// Loads a complete email from a .EML file. (EML files are simply RFC822 MIME text
	// files.)
	// 
	// Note: This replaces the entire contents of the email object, including the To/CC
	// recipients.
	// 
	bool LoadEml(const char *mimePath);


	// Loads the email object from a completed asynchronous task.
	bool LoadTaskResult(CkTask &task);


	// Loads an email with the contents of an XML email file.
	// 
	// Note: This replaces the entire contents of the email object, including the To/CC
	// recipients.
	// 
	bool LoadXml(const char *xmlPath);


	// Loads an email from an XML string (previously obtained by calling the GetXml
	// method). The contents of the calling email object are erased and replaced with
	// the email contained within the XML string.
	bool LoadXmlString(const char *xmlStr);


	// Takes a byte array of multibyte (non-Unicode) data and returns a Unicode
	// Q-Encoded string.
	bool QEncodeBytes(CkByteData &inData, const char *charset, CkString &outEncodedStr);

	// Takes a byte array of multibyte (non-Unicode) data and returns a Unicode
	// Q-Encoded string.
	const char *qEncodeBytes(CkByteData &inData, const char *charset);

	// Takes a Unicode string, converts it to the charset specified in the 2nd
	// parameter, Q-Encodes the converted multibyte data, and returns the encoded
	// Unicode string.
	bool QEncodeString(const char *str, const char *charset, CkString &outEncodedStr);

	// Takes a Unicode string, converts it to the charset specified in the 2nd
	// parameter, Q-Encodes the converted multibyte data, and returns the encoded
	// Unicode string.
	const char *qEncodeString(const char *str, const char *charset);

	// Removes the Nth message/rfc822 sub-part of the email. Indexing begins at 0.
	void RemoveAttachedMessage(int idx);


	// Removes all message/rfc822 sub-parts of the email object.
	void RemoveAttachedMessages(void);


	// Removes path information from all attachment filenames.
	void RemoveAttachmentPaths(void);


	// Removes by name all occurrences of a header field.
	void RemoveHeaderField(const char *fieldName);


	// Removes the HTML body from the email (if an HTML body exists).
	void RemoveHtmlAlternative(void);


	// Removes the plain-text body from the email (if a plain-text body exists).
	void RemovePlainTextAlternative(void);


	// Save all the attachments of an email to files in a directory specified by dirPath.
	// The OverwriteExisting property controls whether existing files are allowed to be
	// overwritten.
	// 
	// Note: Email attachment filenames can be renamed or modified prior to saving. The
	// number of attachments is available in the NumAttachments property. An
	// application can loop over the attachments to get the filename for each by
	// calling GetAttachmentFilename(index). Each attachment's filename can be set by
	// calling SetAttachmentFilename(index, newFilename).
	// 
	bool SaveAllAttachments(const char *dirPath);


	// Saves the Nth email attachment to the directory specified by dirPath. The 1st
	// attachment is at index 0. The OverwriteExisting property controls whether
	// existing files are allowed to be overwritten.
	bool SaveAttachedFile(int index, const char *dirPath);


	// Convert this email object to EML and save it to a file.
	bool SaveEml(const char *emlFilePath);


	// Saves the Nth related item to the directory specified by dirPath. (The 1st related
	// item is at index 0) Related content items are typically image or style-sheets
	// embedded within an HTML email. The OverwriteExisting property controls whether
	// existing files are allowed to be overwritten.
	bool SaveRelatedItem(int index, const char *dirPath);


	// Convert this email object to XML and save it to a file.
	bool SaveXml(const char *path);


	// Sets the charset attribute of the content-type header field for a specified
	// attachment. This can be used if the attachment is a text file that contains text
	// in a non us-ascii charset such as Shift_JIS, iso-8859-2, big5, iso-8859-5, etc.
	bool SetAttachmentCharset(int index, const char *charset);


	// Set's an attachment's disposition. The default disposition of an attachment is
	// "attachment". This method is typically called to change the disposition to
	// "inline". The 1st attachment is at index 0.
	bool SetAttachmentDisposition(int index, const char *disposition);


	// Renames a email attachment's filename. The 1st attachment is at index 0.
	bool SetAttachmentFilename(int index, const char *filename);


	// Sets the main body of the email to binary content of any type. The disposition
	// can be an empty string, "inline", or "attachment". If a filename is specified,
	// the disposition must be non-empty because the filename is an attribute of the
	// content-disposition header field.
	bool SetBinaryBody(CkByteData &byteData, const char *contentType, const char *disposition, const char *filename);


	// Allows for a certificate to be explicity provided for decryption. When an email
	// object is loaded via any method, such as LoadEml, SetFromMimeText,
	// SetFromMimeBytes, etc., security layers (signatures and encryption) are
	// automatically unwrapped. This method could be called prior to calling a method
	// that loads the email.
	bool SetDecryptCert(CkCert &cert);


	// Allows for a certificate and private key to be explicity specified for
	// decryption. When an email object is loaded via any method, such as LoadEml,
	// SetFromMimeText, SetFromMimeBytes, etc., security layers (signatures and
	// encryption) are automatically unwrapped. Decryption requires a private key. On
	// Windows-based systems, the private key is often pre-installed and nothing need
	// be done to provide it because Chilkat will automatically find it and use it.
	// However, if not on a Windows system, or if the private key was not
	// pre-installed, then it can be provided by this method, or via the
	// AddPfxSourceFile / AddPfxSourceData methods.
	bool SetDecryptCert2(CkCert &cert, CkPrivateKey &key);


	// Sets the "Date" header field of the email to have the value of the date/time
	// object provided.
	bool SetDt(CkDateTime &dt);


	// Creates a typical email used to send EDIFACT messages. Does the following:
	//     Sets the email body to the EDIFACT message passed in message.
	//     Sets the Content-Transfer-Encoding to Base64.
	//     Set the Content-Type equal to "application/EDIFACT".
	//     Sets the Content-Type header's name attribute to name.
	//     Sets the Content-Disposition equal to "attachment".
	//     Sets the Content-Disposition's "filename" attribute equal to filename.
	//     The EDIFACT message is converted to the charset indicated by charset, and
	//     encoded using Base64 in the email body.
	// The email's subject, recipients, FROM address, and other headers are left
	// unmodified.
	void SetEdifactBody(const char *message, const char *name, const char *filename, const char *charset);


	// Set the encryption certificate to be used in encryption. Use the CreateCS,
	// CertStore, and Cert classes to create a Cert object by either locating a
	// certificate in a certificate store or loading one from a file.
	bool SetEncryptCert(CkCert &cert);


	// Loads an email with the MIME stored in a BinData object.
	// 
	// Note: This replaces the entire contents of the email object, including the To/CC
	// recipients.
	// 
	bool SetFromMimeBd(CkBinData &bindat);


	// Loads the email object with the mimeBytes. If the email object already contained an
	// email, it is entirely replaced. The character encoding (such as "utf-8",
	// "iso-8859-1", etc.) of the bytes is automatically inferred from the content. If
	// for some reason it is not possible to determine the character encoding, the
	// SetFromMimeBytes2 method may be called to explicitly specify the charset.
	bool SetFromMimeBytes(CkByteData &mimeBytes);


	// Loads the email object with the mimeBytes. If the email object already contained an
	// email, it is entirely replaced.
	// 
	// The charset specifies the character encoding of the MIME bytes (such as "utf-8",
	// "iso-8859-1", etc.).
	// 
	bool SetFromMimeBytes2(CkByteData &mimeBytes, const char *charset);


	// Loads an email with the MIME stored in a StringBuilder object.
	// 
	// Note: This replaces the entire contents of the email object, including the To/CC
	// recipients.
	// 
	bool SetFromMimeSb(CkStringBuilder &sb);


	// Loads an email with the contents of a .eml (i.e. MIME) contained in a string.
	// 
	// Note: This replaces the entire contents of the email object, including the To/CC
	// recipients.
	// 
	bool SetFromMimeText(const char *mimeText);


	// Loads an email from an XML string.
	// 
	// Note: This replaces the entire contents of the email object, including the To/CC
	// recipients.
	// 
	bool SetFromXmlText(const char *xmlStr);


	// Sets the HTML body of an email.
	void SetHtmlBody(const char *html);


	// Sets the HTML email body from a byte array containing character data in the
	// specified character set. This method also updates the email "content-type"header
	// to properly reflect the content type of the body.
	bool SetMbHtmlBody(const char *charset, CkByteData &inData);


	// Sets the plain-text email body from a byte array containing character data in
	// the specified character set. This method also updates the email
	// "content-type"header to properly reflect the content type of the body.
	bool SetMbPlainTextBody(const char *charset, CkByteData &inData);


	// Sets the filename for a related item within the email.
	bool SetRelatedFilename(int index, const char *path);


	// Create a pattern/replacement-text pair for mail-merge. When the email is sent
	// via the MailMan's SendEmail method, or any other mail-sending method, the
	// patterns are replaced with the replacement strings during the sending process.
	// Do define multiple replacement patterns, simply call SetReplacePattern once per
	// pattern/replacement string. (Note: The MailMan's RenderToMime method will also
	// do pattern replacements. Methods such as SaveEml or GetMime do not replace
	// patterns.)
	// 
	// Note: Replacement patterns may be placed in any header field, and in both HTML
	// and plain-text email bodies.
	// 
	bool SetReplacePattern(const char *pattern, const char *replaceString);


	// Set the certificate to be used in creating a digital signature. Use the
	// CreateCS, CertStore, and Cert classes to create a Cert object by either locating
	// a certificate in a certificate store or loading one from a file.
	bool SetSigningCert(CkCert &cert);


	// Explicitly sets the certificate and private key to be used for sending digitally
	// signed email. If the certificate's private key is already installed on the
	// computer, then one may simply call SetSigningCert because the Chilkat component
	// will automatically locate and use the corresponding private key (stored in the
	// Windows Protected Store). In most cases, if the digital certificate is already
	// installed w/ private key on the computer, it is not necessary to explicitly set
	// the signing certificate at all. The Chilkat component will automatically locate
	// and use the certificate containing the FROM email address (from the
	// registry-based certificate store where it was installed).
	bool SetSigningCert2(CkCert &cert, CkPrivateKey &key);


	// Sets the body of the email and also sets the Content-Type header field of the
	// contentType. If the email is already multipart/alternative, an additional alternative
	// with the indicated Content-Type will be added. If an alternative with the same
	// Content-Type already exists, it is replaced.
	void SetTextBody(const char *bodyText, const char *contentType);


	// True if the caller email has a UIDL that equals the email passed in the
	// argument.
	bool UidlEquals(CkEmail &e);


	// Unpacks an HTML email into an HTML file and related files (images and style
	// sheets). The links within the HTML are updated to point to the files unpacked
	// and saved to disk.
	// 
	// Note: The UnpackHtml method extracts images that are actually contained within
	// the email itself. If the HTML contains an "IMG" tag that just links to an image
	// on the web (but the image data is not contained locally within the email), then
	// the image is not unpacked (because it would need to be downloaded from the web).
	// 
	bool UnpackHtml(const char *unpackDir, const char *htmlFilename, const char *partsSubdir);


	// Unobfuscates emails by undoing what spammers do to obfuscate email. It removes
	// comments from HTML bodies and unobfuscates hyperlinked URLs.
	void UnSpamify(void);


	// Unzips and replaces any Zip file attachments with the expanded contents. As an
	// example, if an email contained a single Zip file containing 3 GIF image files as
	// an attachment, then after calling this method the email would contain 3 GIF file
	// attachments, and the Zip attachment would be gone.If an email contains multiple
	// Zip file attachments, each Zip is expanded and replaced with the contents.
	bool UnzipAttachments(void);


	// Adds an XML certificate vault to the object's internal list of sources to be
	// searched for certificates and private keys when encrypting/decrypting or
	// signing/verifying. Unlike the AddPfxSourceData and AddPfxSourceFile methods,
	// only a single XML certificate vault can be used. If UseCertVault is called
	// multiple times, only the last certificate vault will be used, as each call to
	// UseCertVault will replace the certificate vault provided in previous calls.
	bool UseCertVault(CkXmlCertVault &vault);


	// Replaces all the attachments of an email with a single Zip file attachment
	// having the filename specified.
	bool ZipAttachments(const char *zipFilename);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
