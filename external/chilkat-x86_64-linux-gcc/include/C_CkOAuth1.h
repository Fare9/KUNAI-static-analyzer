// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkOAuth1_H
#define _C_CkOAuth1_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkOAuth1 CkOAuth1_Create(void);
CK_C_VISIBLE_PUBLIC void CkOAuth1_Dispose(HCkOAuth1 handle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getAuthorizationHeader(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_authorizationHeader(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getBaseString(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_baseString(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getConsumerKey(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putConsumerKey(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_consumerKey(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getConsumerSecret(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putConsumerSecret(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_consumerSecret(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getDebugLogFilePath(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putDebugLogFilePath(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_debugLogFilePath(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getEncodedSignature(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_encodedSignature(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getGeneratedUrl(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_generatedUrl(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getHmacKey(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_hmacKey(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getLastErrorHtml(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_lastErrorHtml(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getLastErrorText(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_lastErrorText(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getLastErrorXml(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_lastErrorXml(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_getLastMethodSuccess(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putLastMethodSuccess(HCkOAuth1 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getNonce(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putNonce(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_nonce(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getOauthMethod(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putOauthMethod(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_oauthMethod(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getOauthUrl(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putOauthUrl(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_oauthUrl(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getOauthVersion(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putOauthVersion(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_oauthVersion(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getQueryString(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_queryString(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getRealm(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putRealm(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_realm(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getSignature(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_signature(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getSignatureMethod(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putSignatureMethod(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_signatureMethod(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getTimestamp(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putTimestamp(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_timestamp(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getToken(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putToken(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_token(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getTokenSecret(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putTokenSecret(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_tokenSecret(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getUncommonOptions(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putUncommonOptions(HCkOAuth1 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_uncommonOptions(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_getUtf8(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putUtf8(HCkOAuth1 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_getVerboseLogging(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC void CkOAuth1_putVerboseLogging(HCkOAuth1 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkOAuth1_getVersion(HCkOAuth1 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkOAuth1_version(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_AddParam(HCkOAuth1 cHandle, const char *name, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_Generate(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_GenNonce(HCkOAuth1 cHandle, int numBytes);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_GenTimestamp(HCkOAuth1 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_RemoveParam(HCkOAuth1 cHandle, const char *name);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_SaveLastError(HCkOAuth1 cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkOAuth1_SetRsaKey(HCkOAuth1 cHandle, HCkPrivateKey privKey);
#endif
