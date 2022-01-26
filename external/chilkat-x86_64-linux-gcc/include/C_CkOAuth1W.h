// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkOAuth1WH
#define _C_CkOAuth1WH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkOAuth1W CkOAuth1W_Create(void);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_Dispose(HCkOAuth1W handle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getAuthorizationHeader(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_authorizationHeader(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getBaseString(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_baseString(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getConsumerKey(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putConsumerKey(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_consumerKey(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getConsumerSecret(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putConsumerSecret(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_consumerSecret(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getDebugLogFilePath(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putDebugLogFilePath(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_debugLogFilePath(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getEncodedSignature(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_encodedSignature(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getGeneratedUrl(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_generatedUrl(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getHmacKey(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_hmacKey(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getLastErrorHtml(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_lastErrorHtml(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getLastErrorText(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_lastErrorText(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getLastErrorXml(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_lastErrorXml(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1W_getLastMethodSuccess(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putLastMethodSuccess(HCkOAuth1W cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getNonce(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putNonce(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_nonce(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getOauthMethod(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putOauthMethod(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_oauthMethod(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getOauthUrl(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putOauthUrl(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_oauthUrl(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getOauthVersion(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putOauthVersion(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_oauthVersion(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getQueryString(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_queryString(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getRealm(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putRealm(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_realm(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getSignature(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_signature(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getSignatureMethod(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putSignatureMethod(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_signatureMethod(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getTimestamp(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putTimestamp(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_timestamp(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getToken(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putToken(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_token(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getTokenSecret(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putTokenSecret(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_tokenSecret(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getUncommonOptions(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putUncommonOptions(HCkOAuth1W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_uncommonOptions(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1W_getVerboseLogging(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC void  CkOAuth1W_putVerboseLogging(HCkOAuth1W cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkOAuth1W_getVersion(HCkOAuth1W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkOAuth1W_version(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1W_AddParam(HCkOAuth1W cHandle, const wchar_t *name, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1W_Generate(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1W_GenNonce(HCkOAuth1W cHandle, int numBytes);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1W_GenTimestamp(HCkOAuth1W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1W_RemoveParam(HCkOAuth1W cHandle, const wchar_t *name);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1W_SaveLastError(HCkOAuth1W cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1W_SetRsaKey(HCkOAuth1W cHandle, HCkPrivateKeyW privKey);
#endif
