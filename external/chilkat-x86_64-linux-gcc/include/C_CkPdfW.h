// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPdfWH
#define _C_CkPdfWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkPdfW_setAbortCheck(HCkPdfW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkPdfW_setPercentDone(HCkPdfW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkPdfW_setProgressInfo(HCkPdfW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkPdfW_setTaskCompleted(HCkPdfW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkPdfW CkPdfW_Create(void);
CK_C_VISIBLE_PUBLIC void CkPdfW_Dispose(HCkPdfW handle);
CK_C_VISIBLE_PUBLIC void CkPdfW_getDebugLogFilePath(HCkPdfW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPdfW_putDebugLogFilePath(HCkPdfW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPdfW_debugLogFilePath(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC void CkPdfW_getLastErrorHtml(HCkPdfW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPdfW_lastErrorHtml(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC void CkPdfW_getLastErrorText(HCkPdfW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPdfW_lastErrorText(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC void CkPdfW_getLastErrorXml(HCkPdfW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPdfW_lastErrorXml(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_getLastMethodSuccess(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC void  CkPdfW_putLastMethodSuccess(HCkPdfW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkPdfW_getNumPages(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC int CkPdfW_getNumSignatures(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC void CkPdfW_getUncommonOptions(HCkPdfW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPdfW_putUncommonOptions(HCkPdfW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPdfW_uncommonOptions(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_getVerboseLogging(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC void  CkPdfW_putVerboseLogging(HCkPdfW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPdfW_getVersion(HCkPdfW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPdfW_version(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_AddSigningCert(HCkPdfW cHandle, HCkCertW cert);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_GetDss(HCkPdfW cHandle, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_GetSignerCert(HCkPdfW cHandle, int index, HCkCertW cert);
CK_C_VISIBLE_PUBLIC HCkJsonObjectW CkPdfW_LastJsonData(HCkPdfW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_LoadBd(HCkPdfW cHandle, HCkBinDataW pdfData);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_LoadFile(HCkPdfW cHandle, const wchar_t *filePath);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_SaveLastError(HCkPdfW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC void CkPdfW_SetHttpObj(HCkPdfW cHandle, HCkHttpW http);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_SetSignatureJpeg(HCkPdfW cHandle, HCkBinDataW jpgData);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_SetSigningCert(HCkPdfW cHandle, HCkCertW cert);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_SetSigningCert2(HCkPdfW cHandle, HCkCertW cert, HCkPrivateKeyW privateKey);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_SignPdf(HCkPdfW cHandle, HCkJsonObjectW jsonOptions, const wchar_t *outFilePath);
CK_C_VISIBLE_PUBLIC HCkTaskW CkPdfW_SignPdfAsync(HCkPdfW cHandle, HCkJsonObjectW jsonOptions, const wchar_t *outFilePath);
CK_C_VISIBLE_PUBLIC BOOL CkPdfW_VerifySignature(HCkPdfW cHandle, int index, HCkJsonObjectW sigInfo);
#endif
