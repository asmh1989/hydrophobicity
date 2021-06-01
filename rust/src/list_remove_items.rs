pub trait RemoveMultiple<T> {
    /// Remove multiple indices
    fn remove_multiple(&mut self, to_remove: Vec<usize>);

    /// Remove multiple indices with swap_remove, this is faster but reorders elements
    fn swap_remove_multiple(&mut self, to_remove: Vec<usize>);

    /// Remove and return multiple indices
    fn take_multiple(&mut self, to_remove: Vec<usize>) -> Vec<T>;

    /// Remove and return multiple indices, preserving the order specified in the index list
    fn take_multiple_in_order(&mut self, to_remove: &[usize]) -> Vec<T>;

    /// Remove and return multiple indices with swap_remove, this is faster but reorders elements and the results are in reverse order
    fn swap_take_multiple(&mut self, to_remove: Vec<usize>) -> Vec<T>;
}

impl<T> RemoveMultiple<T> for Vec<T> {
    fn remove_multiple(&mut self, mut to_remove: Vec<usize>) {
        to_remove.sort();
        to_remove.reverse();
        for r in to_remove {
            self.remove(r);
        }
    }

    fn swap_remove_multiple(&mut self, mut to_remove: Vec<usize>) {
        to_remove.sort();
        to_remove.reverse();
        for r in to_remove {
            self.swap_remove(r);
        }
    }

    fn take_multiple(&mut self, mut to_remove: Vec<usize>) -> Vec<T> {
        to_remove.sort();
        to_remove.reverse();
        let mut collected = vec![];
        for r in to_remove {
            collected.push(self.remove(r));
        }
        collected.reverse();
        collected
    }

    fn take_multiple_in_order(&mut self, to_remove: &[usize]) -> Vec<T> {
        let mut to_remove = to_remove.iter().copied().enumerate().collect::<Vec<_>>();
        to_remove.sort_by_key(|(_, r)| *r);
        to_remove.reverse();
        let mut collected: Vec<Option<T>> = std::iter::repeat_with(|| None)
            .take(to_remove.len())
            .collect();
        for (i, r) in to_remove {
            collected[i] = Some(self.remove(r));
        }
        collected.into_iter().filter_map(|x| x).collect()
    }

    fn swap_take_multiple(&mut self, mut to_remove: Vec<usize>) -> Vec<T> {
        to_remove.sort();
        to_remove.reverse();
        let mut collected = vec![];
        for r in to_remove {
            collected.push(self.swap_remove(r));
        }
        collected
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn remove_multiple() {
        let mut list = vec!['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        list.remove_multiple(vec![8, 0, 5, 6]);
        assert_eq!(vec!['1', '2', '3', '4', '7', '9'], list);
    }

    #[test]
    fn swap_remove_multiple() {
        let mut list = vec!['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        list.swap_remove_multiple(vec![8, 0, 5, 6]);
        assert_eq!(vec!['9', '1', '2', '3', '4', '7'], list);
    }

    #[test]
    fn take_multiple() {
        let mut list = vec!['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        let taken = list.take_multiple(vec![8, 0, 5, 6]);
        assert_eq!(vec!['1', '2', '3', '4', '7', '9'], list);
        assert_eq!(vec!['0', '5', '6', '8'], taken);
    }

    #[test]
    fn swap_take_multiple() {
        let mut list = vec!['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        let taken = list.swap_take_multiple(vec![8, 0, 5, 6]);
        assert_eq!(vec!['9', '1', '2', '3', '4', '7'], list);
        assert_eq!(vec!['8', '6', '5', '0'], taken);
    }

    #[test]
    fn take_multiple_in_order() {
        let mut list = vec!['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        let taken = list.take_multiple_in_order(&vec![8, 0, 5, 6]);
        assert_eq!(vec!['1', '2', '3', '4', '7', '9'], list);
        assert_eq!(vec!['8', '0', '5', '6'], taken);
    }
}
