// lib/pocketbase.js
import PocketBase from 'pocketbase';

const url = 'https://objectbenchmark.pockethost.io/'
export const client = new PocketBase(url);

export const getImageUrl = (item) => {
	const collectionId = item.collectionId || 'pmlzc10hatw7ufi';
	const fileId = item.id;
	const fileName = item.image;
	return `${url}/api/files/${collectionId}/${fileId}/${fileName}`;
};
export const getImage = (item, fileName) => {
	const collectionId = item.collectionId || 'pmlzc10hatw7ufi';
	const fileId = item.id;
	return `${url}/api/files/${collectionId}/${fileId}/${fileName}`;
};