// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkXmlDSig_H
#define _C_CkXmlDSig_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkXmlDSig CkXmlDSig_Create(void);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_Dispose(HCkXmlDSig handle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_getDebugLogFilePath(HCkXmlDSig cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_putDebugLogFilePath(HCkXmlDSig cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkXmlDSig_debugLogFilePath(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_getExternalRefDirs(HCkXmlDSig cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_putExternalRefDirs(HCkXmlDSig cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkXmlDSig_externalRefDirs(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_getIgnoreExternalRefs(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_putIgnoreExternalRefs(HCkXmlDSig cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_getLastErrorHtml(HCkXmlDSig cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkXmlDSig_lastErrorHtml(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_getLastErrorText(HCkXmlDSig cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkXmlDSig_lastErrorText(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_getLastErrorXml(HCkXmlDSig cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkXmlDSig_lastErrorXml(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_getLastMethodSuccess(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_putLastMethodSuccess(HCkXmlDSig cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkXmlDSig_getNumReferences(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC int CkXmlDSig_getNumSignatures(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC int CkXmlDSig_getRefFailReason(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC int CkXmlDSig_getSelector(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_putSelector(HCkXmlDSig cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_getUtf8(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_putUtf8(HCkXmlDSig cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_getVerboseLogging(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_putVerboseLogging(HCkXmlDSig cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_getVersion(HCkXmlDSig cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkXmlDSig_version(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_getWithComments(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC void CkXmlDSig_putWithComments(HCkXmlDSig cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_CanonicalizeFragment(HCkXmlDSig cHandle, const char *xml, const char *fragmentId, const char *version, const char *prefixList, BOOL withComments, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkXmlDSig_canonicalizeFragment(HCkXmlDSig cHandle, const char *xml, const char *fragmentId, const char *version, const char *prefixList, BOOL withComments);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_CanonicalizeXml(HCkXmlDSig cHandle, const char *xml, const char *version, BOOL withComments, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkXmlDSig_canonicalizeXml(HCkXmlDSig cHandle, const char *xml, const char *version, BOOL withComments);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_GetCerts(HCkXmlDSig cHandle, HCkStringArray sa);
CK_C_VISIBLE_PUBLIC HCkXml CkXmlDSig_GetKeyInfo(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC HCkPublicKey CkXmlDSig_GetPublicKey(HCkXmlDSig cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_IsReferenceExternal(HCkXmlDSig cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_LoadSignature(HCkXmlDSig cHandle, const char *xmlSig);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_LoadSignatureBd(HCkXmlDSig cHandle, HCkBinData binData);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_LoadSignatureSb(HCkXmlDSig cHandle, HCkStringBuilder sbXmlSig);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_ReferenceUri(HCkXmlDSig cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkXmlDSig_referenceUri(HCkXmlDSig cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_SaveLastError(HCkXmlDSig cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_SetHmacKey(HCkXmlDSig cHandle, const char *key, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_SetPublicKey(HCkXmlDSig cHandle, HCkPublicKey pubKey);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_SetRefDataBd(HCkXmlDSig cHandle, int index, HCkBinData binData);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_SetRefDataFile(HCkXmlDSig cHandle, int index, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_SetRefDataSb(HCkXmlDSig cHandle, int index, HCkStringBuilder sb, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_UseCertVault(HCkXmlDSig cHandle, HCkXmlCertVault certVault);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_VerifyReferenceDigest(HCkXmlDSig cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkXmlDSig_VerifySignature(HCkXmlDSig cHandle, BOOL verifyReferenceDigests);
#endif
