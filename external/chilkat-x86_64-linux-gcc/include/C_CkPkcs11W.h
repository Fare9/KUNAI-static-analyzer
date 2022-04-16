// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPkcs11WH
#define _C_CkPkcs11WH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkPkcs11W CkPkcs11W_Create(void);
CK_C_VISIBLE_PUBLIC void CkPkcs11W_Dispose(HCkPkcs11W handle);
CK_C_VISIBLE_PUBLIC void CkPkcs11W_getDebugLogFilePath(HCkPkcs11W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPkcs11W_putDebugLogFilePath(HCkPkcs11W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPkcs11W_debugLogFilePath(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11W_getLastErrorHtml(HCkPkcs11W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPkcs11W_lastErrorHtml(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11W_getLastErrorText(HCkPkcs11W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPkcs11W_lastErrorText(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11W_getLastErrorXml(HCkPkcs11W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPkcs11W_lastErrorXml(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_getLastMethodSuccess(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC void  CkPkcs11W_putLastMethodSuccess(HCkPkcs11W cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkPkcs11W_getNumCerts(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11W_getSharedLibPath(HCkPkcs11W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPkcs11W_putSharedLibPath(HCkPkcs11W cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPkcs11W_sharedLibPath(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_getVerboseLogging(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC void  CkPkcs11W_putVerboseLogging(HCkPkcs11W cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPkcs11W_getVersion(HCkPkcs11W cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPkcs11W_version(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_CloseSession(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_Discover(HCkPkcs11W cHandle, BOOL onlyTokensPresent, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_FindAllCerts(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_FindCert(HCkPkcs11W cHandle, const wchar_t *certPart, const wchar_t *partValue, HCkCertW cert);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_GetCert(HCkPkcs11W cHandle, int index, HCkCertW cert);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_Initialize(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_InitPin(HCkPkcs11W cHandle, const wchar_t *pin);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_InitToken(HCkPkcs11W cHandle, int slotId, const wchar_t *pin, const wchar_t *tokenLabel);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_Login(HCkPkcs11W cHandle, int userType, const wchar_t *pin);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_Logout(HCkPkcs11W cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_OpenSession(HCkPkcs11W cHandle, int slotId, BOOL readWrite);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_SaveLastError(HCkPkcs11W cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11W_SetPin(HCkPkcs11W cHandle, const wchar_t *oldPin, const wchar_t *newPin);
#endif
