// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkJwt_H
#define _C_CkJwt_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkJwt CkJwt_Create(void);
CK_C_VISIBLE_PUBLIC void CkJwt_Dispose(HCkJwt handle);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_getAutoCompact(HCkJwt cHandle);
CK_C_VISIBLE_PUBLIC void CkJwt_putAutoCompact(HCkJwt cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJwt_getDebugLogFilePath(HCkJwt cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkJwt_putDebugLogFilePath(HCkJwt cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkJwt_debugLogFilePath(HCkJwt cHandle);
CK_C_VISIBLE_PUBLIC void CkJwt_getLastErrorHtml(HCkJwt cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJwt_lastErrorHtml(HCkJwt cHandle);
CK_C_VISIBLE_PUBLIC void CkJwt_getLastErrorText(HCkJwt cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJwt_lastErrorText(HCkJwt cHandle);
CK_C_VISIBLE_PUBLIC void CkJwt_getLastErrorXml(HCkJwt cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJwt_lastErrorXml(HCkJwt cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_getLastMethodSuccess(HCkJwt cHandle);
CK_C_VISIBLE_PUBLIC void CkJwt_putLastMethodSuccess(HCkJwt cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_getUtf8(HCkJwt cHandle);
CK_C_VISIBLE_PUBLIC void CkJwt_putUtf8(HCkJwt cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_getVerboseLogging(HCkJwt cHandle);
CK_C_VISIBLE_PUBLIC void CkJwt_putVerboseLogging(HCkJwt cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJwt_getVersion(HCkJwt cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJwt_version(HCkJwt cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_CreateJwt(HCkJwt cHandle, const char *header, const char *payload, const char *password, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJwt_createJwt(HCkJwt cHandle, const char *header, const char *payload, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_CreateJwtPk(HCkJwt cHandle, const char *header, const char *payload, HCkPrivateKey key, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJwt_createJwtPk(HCkJwt cHandle, const char *header, const char *payload, HCkPrivateKey key);
CK_C_VISIBLE_PUBLIC int CkJwt_GenNumericDate(HCkJwt cHandle, int numSecOffset);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_GetHeader(HCkJwt cHandle, const char *token, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJwt_getHeader(HCkJwt cHandle, const char *token);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_GetPayload(HCkJwt cHandle, const char *token, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJwt_getPayload(HCkJwt cHandle, const char *token);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_IsTimeValid(HCkJwt cHandle, const char *jwt, int leeway);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_SaveLastError(HCkJwt cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_VerifyJwt(HCkJwt cHandle, const char *token, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkJwt_VerifyJwtPk(HCkJwt cHandle, const char *token, HCkPublicKey key);
#endif
