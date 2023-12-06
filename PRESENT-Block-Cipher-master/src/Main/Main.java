package Main;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
public class Main {

	static byte[] staticPlainText = {0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf};//????
	static byte[] staticKey = {0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf};//???
	static byte[] cipherText = new byte[16];
	static byte[] staticCopyedKey = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; //??????????????????
	static byte[] staticEncryptText = {3, 3, 3, 3, 13, 12, 13, 3, 2, 1, 3, 2, 1, 0, 13, 2};//????
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.arraycopy(staticKey, 0, staticCopyedKey, 0,staticKey.length);
		presentEncrypt.Encrypt(staticPlainText, staticKey);
		presentDecrypt.Decrypt(staticEncryptText, staticCopyedKey);
		byte[] alpha = new byte[]{1,2,3,0};
		System.out.println(getActivateNumber(alpha));
		//byte[] key = {6,13,
		// 10,11,3,1,7,4,4,15,4,1,13,7,0,0,8,7,5,9};
		//byte[] key = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		//Byte[] bkey = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		//ArrayList<Byte[]> roundKeyList = new ArrayList<Byte[]>();
		//Byte[] bkey = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
		//roundKeyList.add(bkey);
		//for(int i = 0;i<31;i++){
		//System.out.println("key:"+Arrays.toString(key));
		//key = presentEncrypt.UpdataKeys(key, i+1);
		//bkey = new Byte[20];
			//for(int j = 0;j<20;j++) {
				//bkey[j] = key[j];
			//}
		//roundKeyList.add(bkey);
		//}
		
		//System.out.println("1");
		byte[] byteArray = {5, 3, 8, 2, 7};

		// 获取降序排列的下标数组
		int[] sortedIndices = sortAndGetIndices(byteArray);

		// 打印结果
		for (int index : sortedIndices) {
			System.out.println("Index: " + index + ", Value: " + byteArray[index]);
		}
	}
	private static int[] sortAndGetIndices(byte[] array) {
		// 创建一个包含原始下标和对应值的数组
		Integer[] indices = new Integer[array.length];
		for (int i = 0; i < array.length; i++) {
			indices[i] = i;
		}

		// 使用自定义比较器，根据元素值进行降序排序
		Arrays.sort(indices, Comparator.comparingInt(index -> array[(int) index]).reversed());

		// 将 Integer 数组转换为 int 数组
		return Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
	}
	private static int getActivateNumber(byte[] alpha){
		int count = 0;
		for (int i = 0; i<alpha.length;i++) {
			if (alpha[i] != 0){
				count++;
			}
		}
		return count;
	}
}

