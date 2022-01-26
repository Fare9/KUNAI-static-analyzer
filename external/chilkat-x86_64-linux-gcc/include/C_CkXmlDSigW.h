// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkXmlDSigWH
#define _C_CkXmlDSigWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkXmlDSigW CkXmlDSigW_Create(void);
CK_C_VISIBLE_PUBLIC void CkXmlDSigW_Dispose(HCkXmlDSigW handle);
CK_C_VISIBLE_PUBLIC void CkXmlDSigW_getDebugLogFilePath(HCkXmlDSigW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkXmlDSigW_putDebugLogFilePath(HCkXmlDSigW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkXmlDSigW_debugLogFilePath(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSigW_getExternalRefDirs(HCkXmlDSigW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkXmlDSigW_putExternalRefDirs(HCkXmlDSigW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkXmlDSigW_externalRefDirs(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_getIgnoreExternalRefs(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC void  CkXmlDSigW_putIgnoreExternalRefs(HCkXmlDSigW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkXmlDSigW_getLastErrorHtml(HCkXmlDSigW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkXmlDSigW_lastErrorHtml(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSigW_getLastErrorText(HCkXmlDSigW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkXmlDSigW_lastErrorText(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSigW_getLastErrorXml(HCkXmlDSigW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkXmlDSigW_lastErrorXml(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_getLastMethodSuccess(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC void  CkXmlDSigW_putLastMethodSuccess(HCkXmlDSigW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkXmlDSigW_getNumReferences(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC int CkXmlDSigW_getNumSignatures(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC int CkXmlDSigW_getRefFailReason(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC int CkXmlDSigW_getSelector(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC void  CkXmlDSigW_putSelector(HCkXmlDSigW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_getVerboseLogging(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC void  CkXmlDSigW_putVerboseLogging(HCkXmlDSigW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkXmlDSigW_getVersion(HCkXmlDSigW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkXmlDSigW_version(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_getWithComments(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC void  CkXmlDSigW_putWithComments(HCkXmlDSigW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_CanonicalizeFragment(HCkXmlDSigW cHandle, const wchar_t *xml, const wchar_t *fragmentId, const wchar_t *version, const wchar_t *prefixList, BOOL withComments, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkXmlDSigW_canonicalizeFragment(HCkXmlDSigW cHandle, const wchar_t *xml, const wchar_t *fragmentId, const wchar_t *version, const wchar_t *prefixList, BOOL withComments);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_CanonicalizeXml(HCkXmlDSigW cHandle, const wchar_t *xml, const wchar_t *version, BOOL withComments, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkXmlDSigW_canonicalizeXml(HCkXmlDSigW cHandle, const wchar_t *xml, const wchar_t *version, BOOL withComments);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_GetCerts(HCkXmlDSigW cHandle, HCkStringArrayW sa);
CK_C_VISIBLE_PUBLIC HCkXmlW CkXmlDSigW_GetKeyInfo(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC HCkPublicKeyW CkXmlDSigW_GetPublicKey(HCkXmlDSigW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_IsReferenceExternal(HCkXmlDSigW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_LoadSignature(HCkXmlDSigW cHandle, const wchar_t *xmlSig);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_LoadSignatureBd(HCkXmlDSigW cHandle, HCkBinDataW binData);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_LoadSignatureSb(HCkXmlDSigW cHandle, HCkStringBuilderW sbXmlSig);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_ReferenceUri(HCkXmlDSigW cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkXmlDSigW_referenceUri(HCkXmlDSigW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_SaveLastError(HCkXmlDSigW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_SetHmacKey(HCkXmlDSigW cHandle, const wchar_t *key, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_SetPublicKey(HCkXmlDSigW cHandle, HCkPublicKeyW pubKey);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_SetRefDataBd(HCkXmlDSigW cHandle, int index, HCkBinDataW binData);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_SetRefDataFile(HCkXmlDSigW cHandle, int index, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_SetRefDataSb(HCkXmlDSigW cHandle, int index, HCkStringBuilderW sb, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_UseCertVault(HCkXmlDSigW cHandle, HCkXmlCertVaultW certVault);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_VerifyReferenceDigest(HCkXmlDSigW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSigW_VerifySignature(HCkXmlDSigW cHandle, BOOL verifyReferenceDigests);
#endif
