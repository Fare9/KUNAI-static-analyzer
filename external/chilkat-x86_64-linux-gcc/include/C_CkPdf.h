// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPdf_H
#define _C_CkPdf_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkPdf_setAbortCheck(HCkPdf cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkPdf_setPercentDone(HCkPdf cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkPdf_setProgressInfo(HCkPdf cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkPdf_setTaskCompleted(HCkPdf cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkPdf_setAbortCheck2(HCkPdf cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkPdf_setPercentDone2(HCkPdf cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkPdf_setProgressInfo2(HCkPdf cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkPdf_setTaskCompleted2(HCkPdf cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkPdf_setExternalProgress(HCkPdf cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkPdf_setCallbackContext(HCkPdf cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkPdf CkPdf_Create(void);
CK_C_VISIBLE_PUBLIC void CkPdf_Dispose(HCkPdf handle);
CK_C_VISIBLE_PUBLIC void CkPdf_getDebugLogFilePath(HCkPdf cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPdf_putDebugLogFilePath(HCkPdf cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPdf_debugLogFilePath(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC void CkPdf_getLastErrorHtml(HCkPdf cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPdf_lastErrorHtml(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC void CkPdf_getLastErrorText(HCkPdf cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPdf_lastErrorText(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC void CkPdf_getLastErrorXml(HCkPdf cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPdf_lastErrorXml(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_getLastMethodSuccess(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC void CkPdf_putLastMethodSuccess(HCkPdf cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkPdf_getNumPages(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC int CkPdf_getNumSignatures(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC void CkPdf_getUncommonOptions(HCkPdf cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPdf_putUncommonOptions(HCkPdf cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPdf_uncommonOptions(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_getUtf8(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC void CkPdf_putUtf8(HCkPdf cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_getVerboseLogging(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC void CkPdf_putVerboseLogging(HCkPdf cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPdf_getVersion(HCkPdf cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPdf_version(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_AddSigningCert(HCkPdf cHandle, HCkCert cert);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_GetDss(HCkPdf cHandle, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_GetSignerCert(HCkPdf cHandle, int index, HCkCert cert);
CK_C_VISIBLE_PUBLIC HCkJsonObject CkPdf_LastJsonData(HCkPdf cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_LoadBd(HCkPdf cHandle, HCkBinData pdfData);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_LoadFile(HCkPdf cHandle, const char *filePath);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_SaveLastError(HCkPdf cHandle, const char *path);
CK_C_VISIBLE_PUBLIC void CkPdf_SetHttpObj(HCkPdf cHandle, HCkHttp http);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_SetSignatureJpeg(HCkPdf cHandle, HCkBinData jpgData);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_SetSigningCert(HCkPdf cHandle, HCkCert cert);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_SetSigningCert2(HCkPdf cHandle, HCkCert cert, HCkPrivateKey privateKey);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_SignPdf(HCkPdf cHandle, HCkJsonObject jsonOptions, const char *outFilePath);
CK_C_VISIBLE_PUBLIC HCkTask CkPdf_SignPdfAsync(HCkPdf cHandle, HCkJsonObject jsonOptions, const char *outFilePath);
CK_C_VISIBLE_PUBLIC BOOL CkPdf_VerifySignature(HCkPdf cHandle, int index, HCkJsonObject sigInfo);
#endif
