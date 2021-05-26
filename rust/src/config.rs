use std::{collections::HashMap, sync::Mutex};

use log::warn;
use once_cell::sync::Lazy;

const VDW_RADII: Lazy<Mutex<HashMap<&str, f64>>> = Lazy::new(|| {
    let m = [
        ("C", 1.7),
        ("CA", 1.7),
        ("CB", 1.7),
        ("N", 1.55),
        ("O", 1.52),
        ("CD", 1.7),
        ("CG", 1.7),
        ("CZ", 1.7),
        ("NE", 1.55),
        ("NH1", 1.55),
        ("NH2", 1.55),
        ("ND2", 1.55),
        ("OD1", 1.52),
        ("OD2", 1.52),
        ("SG", 1.8),
        ("NE2", 1.55),
        ("OE1", 1.52),
        ("OE2", 1.52),
        ("CD2", 1.7),
        ("CE1", 1.7),
        ("ND1", 1.55),
        ("CD1", 1.7),
        ("CG1", 1.7),
        ("CG2", 1.7),
        ("OXT", 1.52),
        ("CE", 1.7),
        ("NZ", 1.55),
        ("SD", 1.8),
        ("CE2", 1.7),
        ("OG", 1.52),
        ("OG1", 1.52),
        ("CE3", 1.7),
        ("CH2", 1.7),
        ("CZ2", 1.7),
        ("CZ3", 1.7),
        ("NE1", 1.55),
        ("OH", 1.52),
        ("H1", 1.2),
        ("H2", 1.2),
        ("P", 1.8),
        ("O1P", 1.52),
        ("O2P", 1.52),
        ("O3P", 1.52),
        ("F", 1.47),
        ("MN", 0.8),
        ("NA", 0.8),
        ("ZN", 0.8),
    ]
    .iter()
    .cloned()
    .collect();

    Mutex::new(m)
});

pub fn get_vdw_radii(key: &str) -> f64 {
    let f = VDW_RADII.lock().unwrap().get(key).map_or_else(
        || {
            warn!("key : {} not found vdw radii map", &key);
            0.
        },
        |p| *p,
    );
    f
}

pub fn init_config() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();
}

mod tests {

    #[test]
    fn test_vdw() {
        super::init_config();
        assert_eq!(super::get_vdw_radii("C"), 1.7f64);
        assert_ne!(super::get_vdw_radii("O"), 1.7f64);
        assert_eq!(super::get_vdw_radii("1"), 0.);
    }
}
