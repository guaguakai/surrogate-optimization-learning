import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class UserItemData:
    def __init__(self, user_dict, item_dict, users, items, user_features):
        self.user_dict     = user_dict
        self.item_dict     = item_dict
        self.users         = users
        self.items         = items
        self.user_features = user_features

    def getData(self):
        return self.user_dict, self.item_dict, self.users, self.items, self.user_features

    def to(self, device):
        return UserItemData(self.user_dict, self.item_dict, self.users.to(device), self.items.to(device), self.user_features.to(device))

class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings, item_size=200, user_chunk_size=200, feature_size=200, num_samples=1000000):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        # explicit feedback using _normalize and implicit using _binarize
        self.preprocess_ratings = self._normalize(ratings)
        # self.preprocess_ratings = self._binarize(ratings)

        self.user_list, self.item_list = ratings['userId'].unique(), ratings['itemId'].unique()
        self.num_users, self.num_items = len(self.user_list), len(self.item_list)
        random.shuffle(self.user_list)
        random.shuffle(self.item_list)

        self.user_list, self.item_list = self.user_list[:num_samples*user_chunk_size], self.item_list[:feature_size+item_size]
        self.user_pool, self.item_pool = set(self.user_list), set(self.item_list)

        self.preprocess_ratings = self.preprocess_ratings[(self.preprocess_ratings['userId'].isin(self.user_pool)) & (self.preprocess_ratings['itemId'].isin(self.item_pool))]

        self.item_chunks = [self.item_list[:feature_size], self.item_list[feature_size:feature_size+item_size]] # existing items and new items
        self.user_chunks = [self.user_list[i*user_chunk_size: (i+1)*user_chunk_size] for i in range((len(self.user_list)) // user_chunk_size)] # ignoring the remaining

        self.indices = list(range(len(self.user_chunks)))
        self.train_user_indices    = self.indices[:int(0.7 * len(self.user_chunks))]
        self.validate_user_indices = self.indices[int(0.7 * len(self.user_chunks)): int(0.7 * len(self.user_chunks)) + int(0.1 * len(self.user_chunks))]
        self.test_user_indices     = self.indices[int(0.7 * len(self.user_chunks)) + int(0.1 * len(self.user_chunks)):]

        # create negative item samples for NCF learning
        # print('Generating negative samples...')
        self.negatives = self._sample_negative(ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 32))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def instance_a_train_loader_chunk(self, num_negatives):
        """instance train loader for one training epoch"""
        train_list, validate_list, test_list = [], [], []

        all_ratings = self.preprocess_ratings
        all_ratings = pd.merge(self.preprocess_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        all_ratings['negatives'] = all_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        itemset_feature = self.item_chunks[0]
        item_feature_dict = {k: v for v, k in enumerate(itemset_feature)}
        itemset = self.item_chunks[1]
        item_dict = {k: v for v, k in enumerate(itemset)}
        for userset_id, userset in enumerate(self.user_chunks):
            users, items, ratings = [], [], []
            rating_chunk  = all_ratings[(all_ratings['userId'].isin(userset)) & (all_ratings['itemId'].isin(itemset))]
            for row in rating_chunk.itertuples():
                users.append(int(row.userId))
                items.append(int(row.itemId))
                ratings.append(float(row.rating))

                # valid_negatives = set(row.negative_items).intersection(set(itemset))
                # negative_items = random.sample(valid_negatives, min(num_negatives, len(valid_negatives)))
                negative_items = set(row.negatives).intersection(set(itemset))
                for negative_item in negative_items:
                    users.append(int(row.userId))
                    items.append(int(negative_item))
                    ratings.append(float(0))  # negative samples get 0 rating
            indices = list(range(len(users)))

            user_dict = {k: v for v, k in enumerate(userset)}
            c_target  = torch.zeros(1, len(itemset), len(userset))
            for user_id, item_id, rating in zip(users, items, ratings):
                c_target[0, item_dict[item_id], user_dict[user_id]] = rating
            random.shuffle(indices)
            users, items = torch.LongTensor(users), torch.LongTensor(items)

            # retriving the features of each user
            feature_chunk = all_ratings[(all_ratings['userId'].isin(userset)) & (all_ratings['itemId'].isin(itemset_feature))]
            user_features  = torch.zeros(len(userset), len(itemset_feature))
            for row in feature_chunk.itertuples():
                user_features[user_dict[int(row.userId)], item_feature_dict[int(row.itemId)]] = row.rating

            instance_data = (UserItemData(user_dict, item_dict, users[indices], items[indices], user_features[[user_dict[userId.item()] for userId in users[indices]]]), c_target)
            if userset_id in self.test_user_indices:
                test_list.append(instance_data)
            elif userset_id in self.validate_user_indices:
                validate_list.append(instance_data)
            else:
                train_list.append(instance_data)

        return train_list, validate_list, test_list

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
