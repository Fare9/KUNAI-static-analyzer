// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkTrustedRootsWH
#define _C_CkTrustedRootsWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_setAbortCheck(HCkTrustedRootsW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_setPercentDone(HCkTrustedRootsW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_setProgressInfo(HCkTrustedRootsW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_setTaskCompleted(HCkTrustedRootsW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkTrustedRootsW CkTrustedRootsW_Create(void);
CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_Dispose(HCkTrustedRootsW handle);
CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_getDebugLogFilePath(HCkTrustedRootsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkTrustedRootsW_putDebugLogFilePath(HCkTrustedRootsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTrustedRootsW_debugLogFilePath(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_getLastErrorHtml(HCkTrustedRootsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTrustedRootsW_lastErrorHtml(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_getLastErrorText(HCkTrustedRootsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTrustedRootsW_lastErrorText(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_getLastErrorXml(HCkTrustedRootsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTrustedRootsW_lastErrorXml(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_getLastMethodSuccess(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkTrustedRootsW_putLastMethodSuccess(HCkTrustedRootsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkTrustedRootsW_getNumCerts(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_getRejectSelfSignedCerts(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkTrustedRootsW_putRejectSelfSignedCerts(HCkTrustedRootsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_getTrustSystemCaRoots(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkTrustedRootsW_putTrustSystemCaRoots(HCkTrustedRootsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_getVerboseLogging(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkTrustedRootsW_putVerboseLogging(HCkTrustedRootsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkTrustedRootsW_getVersion(HCkTrustedRootsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTrustedRootsW_version(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_Activate(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_AddCert(HCkTrustedRootsW cHandle, HCkCertW cert);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_AddJavaKeyStore(HCkTrustedRootsW cHandle, HCkJavaKeyStoreW keystore);
CK_C_VISIBLE_PUBLIC HCkTaskW CkTrustedRootsW_AddJavaKeyStoreAsync(HCkTrustedRootsW cHandle, HCkJavaKeyStoreW keystore);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_Deactivate(HCkTrustedRootsW cHandle);
CK_C_VISIBLE_PUBLIC HCkCertW CkTrustedRootsW_GetCert(HCkTrustedRootsW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_LoadCaCertsPem(HCkTrustedRootsW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC HCkTaskW CkTrustedRootsW_LoadCaCertsPemAsync(HCkTrustedRootsW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_LoadTaskCaller(HCkTrustedRootsW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkTrustedRootsW_SaveLastError(HCkTrustedRootsW cHandle, const wchar_t *path);
#endif
