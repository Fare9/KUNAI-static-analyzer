/*
 * virtualized class (1 virtualized method)
 * identifiers were renamed for clarity
 */

public final class VClass {
    private int keylen;
    private byte[] sbox;
    private byte[] tbox;
    private static int guard0;
    private static int guard1;

    /*
     * the constructor is NOT virtualized
     * we recognize a key scheduling algorithm
     */
    public VClass(byte[] key) {
        this.sbox = new byte[0x100];
        this.tbox = new byte[0x100];
        if (key.length <= 0 || key.length > 0x100) {
            throw new IllegalArgumentException("illegal key length");
        }

        this.keylen = key.length;
        int i;
        for (i = 0; i < 0x100; ++i) {
            this.sbox[i] = (byte) i;
            this.tbox[i] = key[i % this.keylen];
        }

        int j = 0;
        int k = 0;
        while (j < 0x100) {
            k = this.sbox[j] + k + this.tbox[j] & 0xFF;
            byte v0 = this.sbox[k];
            this.sbox[k] = this.sbox[j];
            this.sbox[j] = v0;
            ++j;
        }
    }

    // virtualized method, likely doing encryption/decryption
    public final byte[] d(byte[] arg10) {
        PCodeVM vm = new PCodeVM(this, arg10);
        int[] pcode = { -1, 1, 2, 3, 4, 5, -2, 7, -3, -4, 9, 10, 13, -5, 14, 13, -6, -7, -8, 16, -9, 17, -9, 18, -10,
                19, -11, 20, 21, 22, 23, 24, 25, -12, 26, -10, -13, 27, 28, 13, -14, 14, 13, -15, -16, -17, -18, 14, 14,
                13, 0x1F, -19, 14, 14, 13, 0x1F, -12, -8, -20, -17, 0x20, 33, 34, 35, 36, -21, 39, 40, 41, 42, 43, -21,
                44, 35, 36, -21, 45, 46, 0x2F, 36, -21, 46, 0x30, 36, -21, 46, 49, -21, 50, 41, 51, 36, -21, 22, 52, 53,
                13, -22, -23, -24, 54, -25, -26, -27 };
        int idx = 0;
        while (true) {
            int idx1 = idx + 1;
            switch (vm.exec(pcode[idx])) {
                case -27: {
                    idx = 34;
                    break;
                }
                case -26: {
                    idx = 23;
                    break;
                }
                case -25: {
                    vm.exec(55);
                    if (vm.intLoaded == 0) {
                        idx = 103;
                        break;
                    }

                    idx = idx1;
                    break;
                }
                case -24: {
                    idx = 19;
                    break;
                }
                case -23: {
                    idx = 21;
                    break;
                }
                case -22: {
                    vm.exec(15);
                    if (vm.intLoaded == 0) {
                        idx = 99;
                        break;
                    }

                    idx = idx1;
                    break;
                }
                case -21: {
                    vm.intStored = 1;
                    vm.exec(11);
                    vm.exec(37);
                    vm.objStored = ((VClass) vm.objLoaded).sbox;
                    vm.exec(38);
                    idx = idx1;
                    break;
                }
                case -20: {
                    vm.exec(30);
                    switch (vm.intLoaded) {
                        case 0: {
                            idx = 9;
                            break;
                        }
                        case 1: {
                            idx = 7;
                            break;
                        }
                    }
                }
                case -19: {
                    idx = 1;
                    break;
                }
                case -18: {
                    vm.exec(30);
                    switch (vm.intLoaded) {
                        case 95: {
                            idx = 27;
                            break;
                        }
                        case 98: {
                            idx = 25;
                            break;
                        }
                    }
                }
                case -17: {
                    idx = 52;
                    break;
                }
                case -16: {
                    idx = 59;
                    break;
                }
                case -15: {
                    vm.exec(29);
                    if (vm.intLoaded == 0) {
                        idx = 45;
                        break;
                    }

                    idx = idx1;
                    break;
                }
                case -14: {
                    vm.intStored = 1;
                    vm.exec(11);
                    vm.exec(12);
                    VClass.guard0 = vm.intLoaded;
                    idx = idx1;
                    break;
                }
                case -13: {
                    vm.intStored = 1;
                    vm.exec(8);
                    idx = idx1;
                    break;
                }
                case -12: {
                    idx = 100;
                    break;
                }
                case -11: {
                    idx = 27;
                    break;
                }
                case -10: {
                    idx = 58;
                    break;
                }
                case -9: {
                    idx = 46;
                    break;
                }
                case -8: {
                    idx = 60;
                    break;
                }
                case -7: {
                    idx = 57;
                    break;
                }
                case -6: {
                    vm.exec(15);
                    if (vm.intLoaded == 0) {
                        idx = 18;
                        break;
                    }

                    idx = idx1;
                    break;
                }
                case -5: {
                    vm.intStored = 1;
                    vm.exec(11);
                    vm.exec(12);
                    VClass.guard1 = vm.intLoaded;
                    idx = idx1;
                    break;
                }
                case -4: {
                    vm.intStored = 0;
                    vm.exec(8);
                    idx = idx1;
                    break;
                }
                case -3: {
                    vm.exec(6);
                    return (byte[]) vm.objLoaded;
                }
                case -2: {
                    idx = 36;
                    break;
                }
                case -1: {
                    idx = 0x2F;
                    break;
                }
                default: {
                    idx = idx1;
                    break;
                }
            }
        }
    }
}
